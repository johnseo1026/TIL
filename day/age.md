# 성별/나이 구분


```python
import cv2 as cv2
import math
import time

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes


faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"


ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']   # 만든사람이 이렇게 하도록 지정해둔것
genderList = ['Male', 'Female']

# Load network
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

frame = cv2.imread("sample1.jpg")
frameFace, bboxes = getFaceBox(faceNet, frame)   # 얼굴찾는 함수
padding = 10
for bbox in bboxes:
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print("Age Output : {}".format(agePreds))
        print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

        label = "{},{}".format(gender, age)
        cv2.putText(frameFace, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)        
        cv2.imwrite("age-gender-out.jpg", frameFace)
```

    Gender : Male, conf = 1.000
    Age Output : [[5.0422677e-05 6.6032046e-03 9.9196303e-01 9.0643363e-05 1.2083673e-03
      4.4209090e-05 2.6494012e-05 1.3531237e-05]]
    Age : (8-12), conf = 0.992
    Gender : Male, conf = 1.000
    Age Output : [[9.8923862e-01 1.0366707e-02 3.6442763e-04 2.6064206e-06 1.5998328e-05
      7.6123233e-06 1.7588355e-06 2.2897334e-06]]
    Age : (0-2), conf = 0.989
    Gender : Female, conf = 0.950
    Age Output : [[1.65868153e-06 1.00786008e-06 1.11874564e-04 3.56943841e-04
      9.99294400e-01 2.22708652e-04 9.56830445e-06 1.78582150e-06]]
    Age : (25-32), conf = 0.999

![age-gender-out](images/age-gender-out.jpg)