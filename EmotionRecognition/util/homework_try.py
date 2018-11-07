import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect(gray, frame):
    faces = face_cascade.detectMultiScale( gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22) 
        smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 15)
            mid = int(round(ex+(ew/2)))
            half = int(round(ey+(eh/2)))
            # PUPIL CENTERS 
            cv2.rectangle(roi_color, (mid,half), (mid,half), (0,0,255), 8)
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_color, (sx,sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    return frame

video_capture = cv2.VideoCapture(0)
while True:
    _ , lastframe = video_capture.read()
    # now we will colorize the last color of the frame
    # color BGR2GRAY is a trick to get the best grayscale using all colors
    gray = cv2.cvtColor(lastframe, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray,lastframe)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

