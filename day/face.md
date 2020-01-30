# 감정인식

파이썬 버전을 낮춰야 함

> conda create -n py36 python=3.6 anaconda
>
> conda activate py36
>
> pip install opencv-python
>
> pip install imutils
>
> dlib
>
> https://pypi.org/simple/dlib/   에서 
>
>  dlib-19.8.1-cp36-cp36m-win_amd64.whl 다운
>
> pip install dlib-19.8.1-cp36-cp36m-win_amd64.whl
>
> pip install --no-dependencies face_recognition
>
> pip install git+https://github.com/ageitgey/face_recognition_models

```python
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import matplotlib.pyplot as plt
import face_recognition
import os
from imutils import paths
```

```python
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
```

```python
frame = cv2.imread("face.jpg")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 0)  
print("Number of faces detected: {}".format(len(rects)))
for r in rects :
    print(r)
```

    Number of faces detected: 1
    [(22, 50) (280, 308)]



```python
for k, d in enumerate(rects):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        shape = predictor(gray, d)
        shape = face_utils.shape_to_np(shape)
        for (x, y) in shape:
          cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
```

    Detection 0: Left: 22 Top: 50 Right: 280 Bottom: 308



```python
cv2.imwrite("out.jpg", frame)
```




    True

