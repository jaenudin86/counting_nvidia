# === VehicleCountingTRT Folder Structure ===
# ├── main.py
# ├── config.yaml
# ├── requirements.txt
# ├── rtsp_server.py (opsional sebagai GStreamer RTSP server)
# └── vehicle_counting.service (untuk autostart systemd)

# === requirements.txt ===
# opencv-python
# numpy
# PyYAML
# pycuda
# pymysql
# tensorrt==8.x (disesuaikan versi di Jetson)
# pygobject

# === config.yaml ===
# video_url: "https://rtmp.ruangkitastudio.com/memfs/xxx.m3u8"
# model_path: "best.engine"
# rtsp_output_port: 8554
# db:
#   host: "103.115.164.119"
#   user: "vehilce_count"
#   password: "HPGnjLGGM6abndKp"
#   database: "vehilce_count"

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
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp, GLib

Gst.init(None)

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

VIDEO_PATH = config["video_url"]
MODEL_PATH = config["model_path"]
PORT = config.get("rtsp_output_port", 8554)

DB_CONFIG = config["db"]

CLASSES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

def async_save_to_db(vehicle_type, direction):
    def db_thread():
        try:
            conn = pymysql.connect(**DB_CONFIG, autocommit=True)
            cursor = conn.cursor()
            now = datetime.now()
            query = "INSERT INTO vehicle_log (vehicle_type, direction, timestamp) VALUES (%s, %s, %s)"
            cursor.execute(query, (vehicle_type, direction, now))
            print(f"[DB] {vehicle_type} {direction} @ {now}")
        except Exception as e:
            print("[DB ERROR]", e)
        finally:
            try:
                cursor.close()
                conn.close()
            except:
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

def postprocess(detections, conf=0.4):
    result = []
    for det in detections:
        if det[4] < conf:
            continue
        cls_id = int(det[5])
        if cls_id in CLASSES:
            box = det[:4] * 640
            result.append((box.astype(int), CLASSES[cls_id]))
    return result

if __name__ == "__main__":
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[ERROR] Tidak bisa buka video")
        exit()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    line_in = ((0, int(height * 0.4)), (width, int(height * 0.4)))
    line_out = ((0, int(height * 0.6)), (width, int(height * 0.6)))
    memory = {}
    total_in, total_out = 0, 0
    in_count = {v: 0 for v in CLASSES.values()}
    out_count = {v: 0 for v in CLASSES.values()}

    model = TRTInference(MODEL_PATH)

    pipeline = Gst.parse_launch(
        f"appsrc name=mysource is-live=true block=true format=3 ! videoconvert ! nvvidconv ! nvv4l2h264enc bitrate=500000 ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port={PORT}"
    )
    appsrc = pipeline.get_by_name("mysource")
    caps = Gst.Caps.from_string(f"video/x-raw,format=BGR,width={width},height={height},framerate={int(fps)}/1")
    appsrc.set_caps(caps)
    pipeline.set_state(Gst.State.PLAYING)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        inp = preprocess(frame)
        dets = model.infer(inp)
        results = postprocess(dets)

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

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

        cv2.line(frame, *line_in, (0,255,0), 2)
        cv2.line(frame, *line_out, (0,0,255), 2)
        cv2.putText(frame, f"IN : {total_in}", (20, height-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"OUT: {total_out}", (20, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        y_offset = 30
        for label in in_count:
            txt = f"{label.upper()} IN: {in_count[label]} OUT: {out_count[label]}"
            cv2.putText(frame, txt, (width-300, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            y_offset += 20

        success, buffer = cv2.imencode(".jpg", frame)
        if success:
            sample = Gst.Sample.new(
                Gst.Buffer.new_wrapped(buffer.tobytes()),
                caps,
                None,
                None
            )
            appsrc.emit("push-sample", sample)

        if os.getenv("DISPLAY"):
            cv2.imshow("Vehicle Counting TensorRT", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            cv2.waitKey(1)

    cap.release()
    pipeline.set_state(Gst.State.NULL)
    cv2.destroyAllWindows()
