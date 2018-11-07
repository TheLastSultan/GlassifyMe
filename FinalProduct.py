# $ edisplay --gl /path/to/image
import cv2
import numpy as np
import random 
global center, glasses, FACE_DETECTED


# lowrly sunglasses_mask:
# [0,26],[57,22],[116,29],[154,36],[208,25], [269,21],[300,26],[298, 49],[294,56], [285,102], [270,177], [233,128], [192,127], [159,60], [150,57], [140,62], [101,127], [64,128], [19,125], [1,47], [0,26]


#Import Spectacles and Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)
    return resized   

def recalculate_mask_ratio(factor, arr= [[0,26],[57,22],[116,29],[154,36],[208,25], [269,21],[300,26],[298, 49],[294,56], [285,102], [270,177], [233,128], [192,127], [159,60], [150,57], [140,62], [101,127], [64,128], [19,125], [1,47], [0,26]]):
  final = []
  for i in arr:
    sub = []
    for j in i:
      factored = int((factor* j ))
      sub.append(factored)
    final.append(sub)
  return final

def drawGlasses(gray, frame):
    canvas = {}
    canvas["face-detected"] = False
    faces = face_cascade.detectMultiScale( gray, 1.3, 5)
    for (x,y,w,h) in faces:
        canvas["face-detected"] = True
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        canvas["center"] = (int(x+(w/2)),int(y+(h*.45)) )
        canvas["resize-ratio"] = (w*.80)/300
        canvas["resize-factor"] = int(w*.80)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # import smiles
        smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_color, (sx,sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    canvas["frame"] = frame
    return canvas


video_capture = cv2.VideoCapture(0)
while True:
    # Import Spectacles
    src = cv2.imread('lowry2.jpg')

    # values
    _ , lastframe = video_capture.read()
    # now we will colorize the last color of the frame
    # color BGR2GRAY is a trick to get the best grayscale using all colors
    gray = cv2.cvtColor(lastframe, cv2.COLOR_BGR2GRAY)
    canvas = drawGlasses(gray,lastframe)

    if canvas["face-detected"] == True:
        # Create Mask 
        src = image_resize(src, canvas["resize-factor"] )
        cv2.imshow("plane", src)
        src_mask = np.zeros(src.shape, src.dtype)
        resize_arr = recalculate_mask_ratio(canvas["resize-ratio"])
        poly = np.array(resize_arr, np.int32)
        cv2.fillPoly(src_mask, [poly], (255, 255, 255))
        output = cv2.seamlessClone(src, canvas["frame"], src_mask, canvas["center"], cv2.NORMAL_CLONE)
        cv2.imshow('Video', output)
    else:
        cv2.imshow('Video', canvas["frame"])


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

