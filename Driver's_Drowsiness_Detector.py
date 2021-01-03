from scipy.spatial import distance as dist
import numpy as np
import time
import dlib
import cv2
import playsound
from threading import Thread
from collections import OrderedDict


EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 100
ALARM_ON = False

COUNTER = 0

def playAlarm(soundfile):
    playsound.playsound(soundfile)

FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h, _ = img.shape

    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

camera = cv2.VideoCapture(0)

predictor_path = 'C:\\Users\\BALARAM\\Desktop\\ostcl\\shape_predictor_68_face_landmarks.dat'

print('[INFO] Downloading face detector and facial landmarks predictor...')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

(lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]

vs = cv2.VideoCapture(0)
fileStream = True
time.sleep(1.0)

# loop over frames from the video stream
while True:
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    ret, frame = camera.read()
    if ret == False:
        print('Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
        break

    frame_resized = resize(frame, width=240)
    frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(frame_gray, 0)
    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(frame_gray, rect)
        shape = shape_to_np(shape)
 
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
 
        # take the minimum eye aspect ratio
        ear = min([leftEAR,rightEAR])

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if not ALARM_ON:

                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    ALARM_ON = True
                    t = Thread(target=playAlarm,args=('alarm.wav',))
                    t.daemon = True
                    t.start()
                cv2.putText(frame, "Warning! Seems he is trying to sleep", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
        else:
            COUNTER = 0
            ALARM_ON = False

        for (x, y) in np.concatenate((leftEye, rightEye), axis=0):
            cv2.circle(frame, (int(x / ratio), int(y / ratio)), 2, (255, 255, 255), -1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
 
cv2.destroyAllWindows()
vs.release()     
