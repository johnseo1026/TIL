

# django

> pip install django  로 설치

![image-20200213094421966](images/image-20200213094421966.png)

> django-admin start 

![image-20200213094617605](images/image-20200213094617605.png)

> django-admin start project mysite   # mysite라는 폴더 생성
>
> cd mysite
>
> python manage.py runserver        # 서버실행
>
> 다시 나와서
>
> python manage.py startapp myapp      # myapp폴더 생성
>
> myapp -> urls.py 생성

> python manage.py migrate  #db 생성코드

* 참고 얼굴인식을 하려면 python 3.6을써야함

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
  > dlib-19.8.1-cp36-cp36m-win_amd64.whl 다운
  >
  > pip install dlib-19.8.1-cp36-cp36m-win_amd64.whl
  >
  > pip install --no-dependencies face_recognition
  >
  > pip install git+https://github.com/ageitgey/face_recognition_models



## mysite

### urls.py

```python
"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('', include('myapp.urls')),
    path('admin/', admin.site.urls),
]
```

### settings.py

![image-20200213104750071](images/image-20200213104750071.png)

> ['templates'] 추가



## myapp

### urls.py



```python
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('test', views.test),
    path('login', views.login),
    path('service', views.service),
    path('logout', views.logout),
    path('uploadimage', views.uploadimage),
]
```



### views.py

```python
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import face


def index(request):
    return HttpResponse("Hello DJango!!!")

def test(request):
    data = {"s":{"img":"test.png"},"list":[1, 2, 3, 4, 5]}
    return render(request, 'template.html', data)

def login(request):
    id = request.GET["id"]
    pwd = request.GET["pwd"]
    if id == pwd:
        request.session["user"] = id
        return redirect("/service")    
    return redirect("/static/login.html")

def logout(request):
    request.session["user"] = ""
    #request.session.pop("user")
    return redirect("/static/login.html")

def service(req):
    if req.session.get("user", "") == "":
        return redirect("/static/login.html")   # 위2 줄은 보안을 위해 해야함 
    html = "Main Service<br>" + req.session.get("user") + "님 감사합니다<a href=/logout>logout</a>"
    return HttpResponse(html)


@csrf_exempt
def uploadimage(request):   

    file = request.FILES['file1']
    filename = file._name    
    fp = open(settings.BASE_DIR + "/static/" + filename, "wb")
    for chunk in file.chunks() :
        fp.write(chunk)
    fp.close()
    
    result = face.facerecognition(settings.BASE_DIR + "/known.bin", settings.BASE_DIR + "/static/" + filename)
    print(result)
    if result != "" : 
        request.session["user"] = result[0]    
        return redirect("/service")
    return redirect("/static/login.html")
```

## face.py

- 얼굴인식을 위해서는 face.py가 필요 

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
import pickle


def facerecognition(model, file) :
    data = pickle.loads(open(model, "rb").read())
    image = cv2.imread(file)
    boxes = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, boxes)
 
    names = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {} 
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1 
            name = max(counts, key=counts.get)
        names.append(name)                             
    return names
  
facerecognition("known.bin", "sana.jpg")
```

## face폴더

face폴더를 만든후 안에 각각의 폴더(이름별로)에 여러 사진을 집어넣은후 

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
import pickle

def imshow(tit, image) :
    plt.title(tit)    
    if len(image.shape) == 3 :
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else :
        plt.imshow(image, cmap="gray")
    plt.show()
    
    
imagePaths = list(paths.list_images("face"))  #face폴더 안에 사진들 넣어줘야함
 
knownEncodings = []
knownNames = []

for (i, imagePath) in enumerate(imagePaths):    
    name = imagePath.split(os.path.sep)[-2]
    print(f"{name}   -   {imagePath}") 
    image = cv2.imread(imagePath)
    boxes = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, boxes) 

    for encoding in encodings:        
        knownEncodings.append(encoding)
        knownNames.append(name)
        

data = {"encodings": knownEncodings, "names": knownNames}
f = open("known.bin", "wb")
f.write(pickle.dumps(data))
f.close()    
```

* 이렇게 known.bin에 학습을 시켜줘야함



## templates

> templates 폴더 생성
>
> ![image-20200213105346537](images/image-20200213105346537.png)

### template.html

```html
안녕   TEST<br>

입력데이터 = {{s.img}}  <br>

{% for l in list %}

    {{l}} <br>

{% endfor %}
```

![image-20200213111620077](images/image-20200213111620077.png)



## static

> static폴더 생성
>
> ![image-20200213112122517](images/image-20200213112122517.png)

> setting.py 에서
>
> ![image-20200213112738629](images/image-20200213112738629.png)
>
> STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static'),]      추가

### login.html

```html
<meta charset="utf-8">
<form action=/login method=get>
    
    id <input type=text  name=id> <br>
    pwd<input type=text  name=pwd> <br>
    <input type=submit  value="로그인">
</form>

<form action = "/uploadimage" method = "POST"  enctype = "multipart/form-data">
    <input type = "file" name = "file1"/><br>
    <input type = "submit" value = "얼굴인증"/>
</form>
```



