import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')

from gi.repository import Gst, GstRtspServer, GLib

Gst.init(None)

class RTSPMediaFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self):
        super(RTSPMediaFactory, self).__init__()
        self.launch_string = (
            'udpsrc port=8554 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! '
            'rtph264depay ! h264parse ! rtph264pay name=pay0 pt=96 config-interval=1'
        )

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

class RTSPServer:
    def __init__(self):
        self.server = GstRtspServer.RTSPServer()
        self.server.set_service("8554")
        factory = RTSPMediaFactory()
        factory.set_shared(True)
        mount_points = self.server.get_mount_points()
        mount_points.add_factory("/live", factory)
        self.server.attach(None)
        print("✅ RTSP stream ready at rtsp://<IP>:8554/live")

if __name__ == '__main__':
    RTSPServer()
    loop = GLib.MainLoop()
    loop.run()
