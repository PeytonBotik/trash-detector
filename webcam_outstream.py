import cv2
from threading import Thread
import gi

# Initialize GStreamer
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)

class VideoStream:
    """Camera object that controls video streaming from the webcam"""
    def __init__(self, resolution=(640,480), framerate=30):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        ret = self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        ret = self.stream.set(cv2.CAP_PROP_FPS, framerate)
        if not ret:
            print("Failed to set camera properties.")
            exit(1)

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

print("Starting script...")

# Initialize video stream
videostream = VideoStream(resolution=(1920,1080), framerate=30).start()
print("Video stream started.")

# Initialize GStreamer pipeline
out_pipeline = (
    'appsrc ! '
    'videoconvert ! '
    'videoscale ! '
    'video/x-raw,width=1920,height=1080 ! '
    'x264enc speed-preset=ultrafast tune=zerolatency ! '
    'rtph264pay config-interval=1 pt=96 ! '
    'udpsink host=192.168.2.2 port=6000 sync=false'
)
out = Gst.parse_launch(out_pipeline)
out.set_state(Gst.State.PLAYING)
print("GStreamer pipeline initialized and playing.")

while True:
    frame = videostream.read()
    if frame is None:
        print("Failed to capture frame from camera. Exiting...")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.tobytes()
    buffer = Gst.Buffer.new_allocate(None, len(frame), None)
    buffer.fill(0, frame)

    appsrc = out.get_by_name('appsrc0')
    appsrc.emit('push-buffer', buffer)

    if cv2.waitKey(1) == ord('q'):
        print("Quit key pressed. Exiting...")
        break

cv2.destroyAllWindows()
videostream.stop()
out.set_state(Gst.State.NULL)
print("Cleaned up resources. Script terminated.")
