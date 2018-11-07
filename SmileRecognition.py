# $ edisplay --gl /path/to/image
import cv2
import numpy as np
import random 
global center, glasses, FACE_DETECTED



#Import Spectacles and Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Glasses
# glasses = cv2.imread('FelixFinal.jpg', -1)
original_mask = [0,26],[57,22],[116,29],[154,36],[208,25], [269,21],[300,26],[298, 49],[294,56], [285,102], [270,177], [233,128], [192,127], [159,60], [150,57], [140,62], [101,127], [64,128], [19,125], [1,47], [0,26]
center = None




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

def recalculate_mask_ratio(factor, arr = original_mask ,):
  final = []
  for i in arr:
    sub = []
    for j in i:
      factored = int((j * factor))
      sub.append(factored)
    final.append(sub)
  return np.array(final, np.int32)


def drawGlasses(gray, frame, glasses):
    FACE_DETECTED = False 
    faces = face_cascade.detectMultiScale( gray, 1.3, 5)
    for (x,y,w,h) in faces:
        FACE_DETECTED = True
        glasses= image_resize( glasses, int(w*.80))
        # glasses = cv2.cvtColor(glasses, cv2.COLOR_BGR2GRAY)

        cv2.imshow('glasses', glasses )
        cv2.imshow('frame', frame)
        center= (int(x+(w/2)),int(y+(h*.3)) )
        print(center)
        glasses_mask = np.zeros(glasses.shape, glasses.dtype)
        print("glasses mask:")
        print(glasses_mask)
        poly = recalculate_mask_ratio((2000/(w*.80)))
        print("poly:")
        print(poly)
        cv2.fillPoly(glasses_mask, [poly], (255, 255, 255))

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        print("done")
        output = cv2.seamlessClone(glasses, frame, glasses_mask, center, cv2.NORMAL_CLONE)
              
    return output

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])

video_capture = cv2.VideoCapture(0)
while True:
    # Glasses
    glasses = cv2.imread('FelixFinal.jpg')
    original_mask = [32,170],[987,151],[2000,175],[2000,830],[1526,1829], [1001,382],[726,846],[146, 828],[1,170]
   
    _ , lastframe = video_capture.read()
    # now we will colorize the last color of the frame
    # color BGR2GRAY is a trick to get the best grayscale using all colors
    gray = cv2.cvtColor(lastframe, cv2.COLOR_BGR2GRAY)
    canvas = drawGlasses(gray,lastframe, glasses)
    
    cv2.imshow('Video', canvas)
    


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

