import cv2
import time

stream_url = 'udp://192.168.2.1:5600' # when connected via BaseStation
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Cannot open stream")
    exit()

ret, frame = cap.read()
if ret:
    cv2.imwrite('photo.jpg', frame)
else:
    print("Failed to capture photo")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('video.mp4', fourcc, 30.0, (1920, 1080))

start_time - time.time()
while int(time.time() - start_time) < 10: # record for 10 seconds
    ret, frame = cap.read()
    if ret:
        out.write(frame)
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()