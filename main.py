# -*- coding: utf-8 -*-

import cv2
import numpy as np
import yaml
import pymysql
import threading
from datetime import datetime
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os
import signal
import gi

# GStreamer init
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp, GLib

Gst.init(None)

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

VIDEO_PATH = config["video_url"]
MODEL_PATH = config["model_path"]
# Gunakan port UDP terpisah untuk pipeline internal; RTSP server tetap di 8554
UDP_PORT = int(config.get("rtsp_udp_port", 5000))
RTSP_PORT = int(config.get("rtsp_output_port", 8554))  # hanya sebagai info

DB_CONFIG = config["db"]

CLASSES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def async_save_to_db(vehicle_type, direction):
    def db_thread():
        try:
            # FIX: gunakan **DB_CONFIG, sebelumnya salah
            conn = pymysql.connect(**DB_CONFIG, autocommit=True)
            cursor = conn.cursor()
            now = datetime.now()
            query = "INSERT INTO vehicle_log (vehicle_type, direction, timestamp) VALUES (%s, %s, %s)"
            cursor.execute(query, (vehicle_type, direction, now))
        except Exception as e:
            print("[DB ERROR]", e)
        finally:
            try:
                cursor.close()
                conn.close()
            except Exception:
                pass
    threading.Thread(target=db_thread, daemon=True).start()


def crossed_line(p1, p2, line_start, line_end):
    def ccw(a, b, c):
        return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
    return ccw(p1, line_start, line_end) != ccw(p2, line_start, line_end) and \
           ccw(p1, p2, line_start) != ccw(p1, p2, line_end)


class TRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        print(f"[INFO] Loading TensorRT engine: {engine_path}")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        for i in range(self.engine.num_bindings):
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.context.get_binding_shape(i)
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(i):
                self.inputs.append((host_mem, device_mem))
            else:
                self.outputs.append((host_mem, device_mem))

    def infer(self, input_data):
        np.copyto(self.inputs[0][0], input_data.ravel())
        cuda.memcpy_htod(self.inputs[0][1], self.inputs[0][0])
        self.context.execute_v2(bindings=self.bindings)
        cuda.memcpy_dtoh(self.outputs[0][0], self.outputs[0][1])
        return self.outputs[0][0].reshape(-1, 6)


def preprocess(frame):
    img = cv2.resize(frame, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return img[np.newaxis, ...]


def nms(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def postprocess(detections, frame_w, frame_h, input_size=640, conf=0.4, iou_thresh=0.45):
    """
    Robust postprocess for TensorRT YOLO-like raw output.
    detections: (N,6) array where columns are [x1, y1, x2, y2, conf, class]
    This function attempts to auto-detect the scale of coordinates and convert boxes to frame size,
    apply confidence threshold and non-maximum suppression (NMS).
    """
    if detections is None or len(detections) == 0:
        return []

    dets = np.array(detections, dtype=np.float32)
    # ensure shape (-1,6)
    if dets.ndim == 1 and dets.size == 6:
        dets = dets.reshape(1,6)
    if dets.shape[1] < 6:
        # unexpected format
        return []

    coords = dets[:,:4]
    scores = dets[:,4]
    classes = dets[:,5].astype(np.int32)

    # quick heuristic to detect coordinate scale
    max_coord = coords.max()
    if max_coord <= 1.0:
        # normalized [0,1] -> map to input_size
        coords = coords * input_size
    elif max_coord > input_size * 1.5:
        # coordinates seem scaled (very large), attempt to normalize by factor
        factor = max_coord / input_size
        coords = coords / factor

    # coords currently in input_size (e.g., 640) space -> convert to frame size
    scale_x = frame_w / float(input_size)
    scale_y = frame_h / float(input_size)
    coords[:,[0,2]] = coords[:,[0,2]] * scale_x
    coords[:,[1,3]] = coords[:,[1,3]] * scale_y

    # convert to x1,y1,x2,y2 ints and clip
    boxes = coords.copy()
    boxes[:,0] = np.clip(boxes[:,0], 0, frame_w-1)
    boxes[:,1] = np.clip(boxes[:,1], 0, frame_h-1)
    boxes[:,2] = np.clip(boxes[:,2], 0, frame_w-1)
    boxes[:,3] = np.clip(boxes[:,3], 0, frame_h-1)

    # filter by confidence and valid boxes (w>1,h>1)
    valid = (scores >= conf) & ((boxes[:,2]-boxes[:,0]) > 2) & ((boxes[:,3]-boxes[:,1]) > 2)
    if not np.any(valid):
        return []

    boxes = boxes[valid]
    scores = scores[valid]
    classes = classes[valid]

    # perform NMS per class
    final = []
    unique_classes = np.unique(classes)
    for cls in unique_classes:
        inds = np.where(classes == cls)[0]
        cls_boxes = boxes[inds]
        cls_scores = scores[inds]
        if len(cls_boxes) == 0:
            continue
        keep = nms(cls_boxes, cls_scores, iou_threshold=iou_thresh)
        for k in keep:
            b = cls_boxes[k].astype(int)
            label = CLASSES.get(int(cls), str(cls))
            final.append((b, label))

    return final


def build_gst_pipeline(width, height, fps):
    # appsrc -> (BGR raw) -> videoconvert -> nvvidconv -> nvv4l2h264enc -> rtph264pay -> udpsink
    launch = (
        f"appsrc name=mysource is-live=true block=true format=time do-timestamp=true ! "
        f"videoconvert ! nvvidconv ! "
        f"nvv4l2h264enc bitrate=800000 insert-sps-pps=true idrinterval=30 ! "
        f"rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port={UDP_PORT} sync=false async=false"
    )
    print("[GST] Launch:", launch)
    pipeline = Gst.parse_launch(launch)
    appsrc = pipeline.get_by_name("mysource")
    # Caps untuk RAW BGR (bukan JPEG). Kita akan push bytes frame.tobytes()
    caps = Gst.Caps.from_string(
        f"video/x-raw,format=BGR,width={width},height={height},framerate={int(fps)}/1"
    )
    appsrc.set_caps(caps)
    # Simpan fps untuk timestamping
    appsrc.set_property("format", Gst.Format.TIME)
    appsrc.set_property("do-timestamp", True)
    return pipeline, appsrc


def push_frame_appsrc(appsrc, frame, fps, frame_count):
    # Buat buffer sebesar frame dan isi data BGR
    data = frame.tobytes()
    buf = Gst.Buffer.new_allocate(None, len(data), None)
    buf.fill(0, data)
    # set PTS/DTS agar pacing benar
    duration = int(1e9 / max(fps, 1))
    buf.pts = buf.dts = frame_count * duration
    buf.duration = duration
    retval = appsrc.emit("push-buffer", buf)
    if retval != Gst.FlowReturn.OK:
        print("[GST] push-buffer not OK:", retval)


if __name__ == "__main__":
    print(f"[INFO] Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[ERROR] Tidak bisa buka video")
        raise SystemExit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_read = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_read if fps_read and fps_read > 0 else 25.0
    print(f"[INFO] Size: {width}x{height} @ {fps:.2f} FPS")

    line_in = ((0, int(height * 0.4)), (width, int(height * 0.4)))
    line_out = ((0, int(height * 0.6)), (width, int(height * 0.6)))
    memory = {}
    total_in, total_out = 0, 0
    in_count = {v: 0 for v in CLASSES.values()}
    out_count = {v: 0 for v in CLASSES.values()}

    model = TRTInference(MODEL_PATH)
    print("[INFO] TensorRT engine ready.")

    pipeline, appsrc = build_gst_pipeline(width, height, fps)
    pipeline.set_state(Gst.State.PLAYING)
    print(f"[INFO] GStreamer pipeline PLAYING -> UDP {UDP_PORT} (RTSP server baca dari sini)")

    # Graceful shutdown
    def handle_sigint(sig, frame):
        print("
[INFO] SIGINT received, sending EOS...")
        try:
            appsrc.emit("end-of-stream")
        except Exception:
            pass
        pipeline.set_state(Gst.State.NULL)
        cap.release()
        cv2.destroyAllWindows()
        os._exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame tidak terbaca, stop.")
            break

        # Inference
        inp = preprocess(frame)
        dets = model.infer(inp)
        results = postprocess(dets)

        # Counting + overlay
        for box, label in results:
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2
            tid = f"{label}_{x_center}_{y_center}"

            if tid not in memory:
                memory[tid] = {"y": y_center}
            else:
                prev_y = memory[tid]["y"]
                memory[tid]["y"] = y_center
                if "counted" not in memory[tid]:
                    prev = (x_center, prev_y)
                    curr = (x_center, y_center)
                    if crossed_line(prev, curr, *line_in):
                        total_in += 1
                        in_count[label] += 1
                        memory[tid]["counted"] = "in"
                        async_save_to_db(label, "in")
                    elif crossed_line(prev, curr, *line_out):
                        total_out += 1
                        out_count[label] += 1
                        memory[tid]["counted"] = "out"
                        async_save_to_db(label, "out")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.line(frame, *line_in, (0, 255, 0), 2)
        cv2.line(frame, *line_out, (0, 0, 255), 2)
        cv2.putText(frame, f"IN : {total_in}", (20, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"OUT: {total_out}", (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        y_offset = 30
        for label in in_count:
            txt = f"{label.upper()} IN: {in_count[label]} OUT: {out_count[label]}"
            cv2.putText(frame, txt, (max(0, width - 320), y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20

        # PUSH RAW BGR (bukan JPEG)
        push_frame_appsrc(appsrc, frame, fps, frame_count)
        frame_count += 1

        if os.getenv("DISPLAY"):
            cv2.imshow("Vehicle Counting TensorRT", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # beri small sleep agar CPU tidak 100%
            cv2.waitKey(1)

    cap.release()
    try:
        appsrc.emit("end-of-stream")
    except Exception:
        pass
    pipeline.set_state(Gst.State.NULL)
    cv2.destroyAllWindows()
