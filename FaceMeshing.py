import cv2
import time
from mediapipe.python.solutions.face_mesh import FaceMesh, FACE_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec


cap = cv2.VideoCapture('videos/7.mp4')
scale = 1
frame_width = int(cap.get(3) * scale)
frame_height = int(cap.get(4) * scale)
# is stored in 'filename.avi' file.
out = cv2.VideoWriter('result.avi',
                      cv2.VideoWriter_fourcc(*'MJPG'),
                      20, (frame_width, frame_height))
mymesh = FaceMesh(static_image_mode=False,
                  max_num_faces=1,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5)
mydrawingspecs = DrawingSpec(thickness=2, circle_radius=1)
pTime = time.time()

while True:
    success, img = cap.read()
    if not success:
        break
    new_h, new_w = int(img.shape[0] * scale), int(img.shape[1] * scale)
    img = cv2.resize(img, (new_w, new_h))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mymesh.process(imgRGB).multi_face_landmarks
    if results:
        for faceLms in results:
            draw_landmarks(img, faceLms, FACE_CONNECTIONS,
                           mydrawingspecs,
                           mydrawingspecs)

    # frame rate
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)),
                (20, 50), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 3)
    out.write(img)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows()
