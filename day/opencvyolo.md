```python
import cv2
import argparse
import numpy as np
import os.path
from matplotlib import pyplot as plt
%matplotlib inline

```


```python
# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

# Load names of classes
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
print(classes)
# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
```

    ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']



```python
# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
```


```python
cap = cv2.VideoCapture('dog.jpg')

hasFrame, frame = cap.read()

#inpWidth =  frame.shape[1]
#inpHeight = frame.shape[0]
# 여기서 inpWidth는 영상의 크기가 아님.
blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

net.setInput(blob)

outs = net.forward(getOutputsNames(net))

postprocess(frame, outs)

t, _ = net.getPerfProfile()
label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
cv2.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

```




    array([[[ 50,  58,  56],
            [ 51,  59,  57],
            [ 53,  61,  59],
            ...,
            [ 31, 102, 119],
            [ 39,  53,  77],
            [ 45,  59,  83]],
    
           [[ 51,  59,  57],
            [ 51,  59,  57],
            [ 52,  60,  58],
            ...,
            [ 21,  86, 104],
            [ 41,  54,  73],
            [ 46,  59,  78]],
    
           [[ 51,  59,  56],
            [ 51,  59,  56],
            [ 52,  60,  57],
            ...,
            [ 10,  64,  83],
            [ 47,  58,  67],
            [ 42,  53,  62]],
    
           ...,
    
           [[179, 167, 160],
            [179, 167, 160],
            [183, 170, 163],
            ...,
            [ 64,  63,  80],
            [ 36,  39,  52],
            [ 48,  51,  64]],
    
           [[181, 170, 161],
            [180, 169, 160],
            [176, 165, 156],
            ...,
            [ 61,  61,  77],
            [ 37,  41,  54],
            [ 52,  56,  69]],
    
           [[177, 166, 157],
            [178, 167, 158],
            [173, 162, 153],
            ...,
            [ 62,  62,  78],
            [ 33,  37,  50],
            [ 35,  39,  52]]], dtype=uint8)




```python
cv2.imwrite("out.jpg", frame) 
img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x12000219248>




![png](images/output_4_1-1580201355801.png)



# 원하는 YOLO계층만 박스치기


```python
import cv2 as cv2
import argparse
import numpy as np
import os.path
import math
from matplotlib import pyplot as plt
%matplotlib inline


def imshow(tit, image) :
    plt.title(tit)    
    if len(image.shape) == 3 :
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else :
        plt.imshow(image, cmap="gray")
    plt.show()
```


```python
# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

# Load names of classes
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
print(classes)

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
```

    ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']



```python
l = net.getLayerNames()
print(l)
print(len(l))

print(net.getUnconnectedOutLayers())


"""
[[200]
 [227]
 [254]]
[200-1   227-1   254-1]   
#[layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
"""

print(l[199])
print(l[226])
print(l[253])
```

    ['conv_0', 'bn_0', 'relu_0', 'conv_1', 'bn_1', 'relu_1', 'conv_2', 'bn_2', 'relu_2', 'conv_3', 'bn_3', 'relu_3', 'shortcut_4', 'conv_5', 'bn_5', 'relu_5', 'conv_6', 'bn_6', 'relu_6', 'conv_7', 'bn_7', 'relu_7', 'shortcut_8', 'conv_9', 'bn_9', 'relu_9', 'conv_10', 'bn_10', 'relu_10', 'shortcut_11', 'conv_12', 'bn_12', 'relu_12', 'conv_13', 'bn_13', 'relu_13', 'conv_14', 'bn_14', 'relu_14', 'shortcut_15', 'conv_16', 'bn_16', 'relu_16', 'conv_17', 'bn_17', 'relu_17', 'shortcut_18', 'conv_19', 'bn_19', 'relu_19', 'conv_20', 'bn_20', 'relu_20', 'shortcut_21', 'conv_22', 'bn_22', 'relu_22', 'conv_23', 'bn_23', 'relu_23', 'shortcut_24', 'conv_25', 'bn_25', 'relu_25', 'conv_26', 'bn_26', 'relu_26', 'shortcut_27', 'conv_28', 'bn_28', 'relu_28', 'conv_29', 'bn_29', 'relu_29', 'shortcut_30', 'conv_31', 'bn_31', 'relu_31', 'conv_32', 'bn_32', 'relu_32', 'shortcut_33', 'conv_34', 'bn_34', 'relu_34', 'conv_35', 'bn_35', 'relu_35', 'shortcut_36', 'conv_37', 'bn_37', 'relu_37', 'conv_38', 'bn_38', 'relu_38', 'conv_39', 'bn_39', 'relu_39', 'shortcut_40', 'conv_41', 'bn_41', 'relu_41', 'conv_42', 'bn_42', 'relu_42', 'shortcut_43', 'conv_44', 'bn_44', 'relu_44', 'conv_45', 'bn_45', 'relu_45', 'shortcut_46', 'conv_47', 'bn_47', 'relu_47', 'conv_48', 'bn_48', 'relu_48', 'shortcut_49', 'conv_50', 'bn_50', 'relu_50', 'conv_51', 'bn_51', 'relu_51', 'shortcut_52', 'conv_53', 'bn_53', 'relu_53', 'conv_54', 'bn_54', 'relu_54', 'shortcut_55', 'conv_56', 'bn_56', 'relu_56', 'conv_57', 'bn_57', 'relu_57', 'shortcut_58', 'conv_59', 'bn_59', 'relu_59', 'conv_60', 'bn_60', 'relu_60', 'shortcut_61', 'conv_62', 'bn_62', 'relu_62', 'conv_63', 'bn_63', 'relu_63', 'conv_64', 'bn_64', 'relu_64', 'shortcut_65', 'conv_66', 'bn_66', 'relu_66', 'conv_67', 'bn_67', 'relu_67', 'shortcut_68', 'conv_69', 'bn_69', 'relu_69', 'conv_70', 'bn_70', 'relu_70', 'shortcut_71', 'conv_72', 'bn_72', 'relu_72', 'conv_73', 'bn_73', 'relu_73', 'shortcut_74', 'conv_75', 'bn_75', 'relu_75', 'conv_76', 'bn_76', 'relu_76', 'conv_77', 'bn_77', 'relu_77', 'conv_78', 'bn_78', 'relu_78', 'conv_79', 'bn_79', 'relu_79', 'conv_80', 'bn_80', 'relu_80', 'conv_81', 'permute_82', 'yolo_82', 'identity_83', 'conv_84', 'bn_84', 'relu_84', 'upsample_85', 'concat_86', 'conv_87', 'bn_87', 'relu_87', 'conv_88', 'bn_88', 'relu_88', 'conv_89', 'bn_89', 'relu_89', 'conv_90', 'bn_90', 'relu_90', 'conv_91', 'bn_91', 'relu_91', 'conv_92', 'bn_92', 'relu_92', 'conv_93', 'permute_94', 'yolo_94', 'identity_95', 'conv_96', 'bn_96', 'relu_96', 'upsample_97', 'concat_98', 'conv_99', 'bn_99', 'relu_99', 'conv_100', 'bn_100', 'relu_100', 'conv_101', 'bn_101', 'relu_101', 'conv_102', 'bn_102', 'relu_102', 'conv_103', 'bn_103', 'relu_103', 'conv_104', 'bn_104', 'relu_104', 'conv_105', 'permute_106', 'yolo_106']
    254
    [[200]
     [227]
     [254]]
    yolo_82
    yolo_94
    yolo_106



```python
# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    
    print(len(boxes))
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)    
    print(indices)    
    
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
```


```python
cap = cv2.VideoCapture('cars.jpg')

hasFrame, frame = cap.read()

#inpWidth =  frame.shape[1]
#inpHeight = frame.shape[0]
# 여기서 inpWidth는 영상의 크기가 아님.
blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
net.setInput(blob)
# 82, 94, 106

#outs = net.forward(getOutputsNames(net))
y_82 = net.forward("yolo_82")
print(y_82.shape)
```

    (507, 85)



```python
frame = cv2.imread("cars.jpg")
frameHeight = frame.shape[0]
frameWidth = frame.shape[1]

blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], True, crop=False)      # 4차원
print(blob.shape)   # 4차원
net.setInput(blob)
# 82, 94, 106
yolo = net.forward("yolo_82")
    
for i in range(len(yolo))     :   
    detection = yolo[i]    
    scores = detection[5:]
    classId = np.argmax(scores)
    confidence = scores[classId]
    if confidence > 0.01:
    #if True:    이거쓰면 다 그려서 지저분해짐    
        center_x = int(detection[0] * frameWidth)
        center_y = int(detection[1] * frameHeight)
        width = int(detection[2] * frameWidth)
        height = int(detection[3] * frameHeight)
        left = int(center_x - width / 2)
        top = int(center_y - height / 2)        
        cv2.rectangle(frame, (left, top), (left+width, top+height), (255, 178, 50), 3)
imshow("", frame)
```

    (1, 3, 416, 416)



![png](images/output_6_1-1580284017696.png)



```python
cells = []
for i in range(len(yolo))     :   
    detection = yolo[i]    
    scores = detection[5:]
    classId = np.argmax(scores)
    cells.append(classId)
```


```python
cells = np.array(cells)
s = int(math.sqrt(len(yolo)/3))

cells = cells.reshape(s,s,3)

img = cells[:,:,0]*30
imshow("", img)
```


![png](images/output_8_0-1580284017697.png)



```python
frame = cv2.imread("dog.jpg")
frameHeight = frame.shape[0]
frameWidth = frame.shape[1]

blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], True, crop=False)      # 4차원
print(blob.shape)   # 4차원
net.setInput(blob)
# 82, 94, 106
yolo = net.forward("yolo_82")
    
for i in range(len(yolo))     :   
    detection = yolo[i]    
    scores = detection[5:]
    classId = np.argmax(scores)
    confidence = scores[classId]
    if confidence > 0.01:
    #if True:    이거쓰면 다 그려서 지저분해짐    
        center_x = int(detection[0] * frameWidth)
        center_y = int(detection[1] * frameHeight)
        width = int(detection[2] * frameWidth)
        height = int(detection[3] * frameHeight)
        left = int(center_x - width / 2)
        top = int(center_y - height / 2)        
        cv2.rectangle(frame, (left, top), (left+width, top+height), (255, 178, 50), 3)
imshow("", frame)
```

    (1, 3, 416, 416)



![png](images/output_9_1-1580284017697.png)



```python
cells = []
for i in range(len(yolo))     :   
    detection = yolo[i]    
    scores = detection[5:]
    classId = np.argmax(scores)
    confidence = scores[classId]
    if confidence > 0.01:
        cells.append(classId+1)
    else :
        cells.append(0)

cells = np.array(cells)
s = int(math.sqrt(len(yolo)/3))

cells = cells.reshape(s,s,3)
```


```python
img = cells[:,:,0]*30
imshow("", img)
```


![png](images/output_11_0-1580284017697.png)


#  영상 만들어서 YOLO


```python
import cv2
import time

cap = cv2.VideoCapture('vtest.avi')
a = int(cap.get(3))    # rlwhs ehd
b = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter("vtest_out.mp4", fourcc, 20.0, (a, b))


for i in range(50) :
    ret, frame = cap.read()         
    blob = cv2.dnn.blobFromImage(frame, 1/255,
            (inpWidth, inpHeight), [0,0,0], True, crop=False)    
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))    
    postprocess(frame, outs)
    #imshow("", frame)
    video.write(frame)    
    cv2.waitKey(20)
video.release()
```

