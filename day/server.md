# 서버구축

> 여기서는 하나 하고 새로운것을 하려면 커널 shutdown 하고 restart를 해야함


```python
import socket
from datetime import datetime
import subprocess
```


```python
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.bind(('localhost', 12345))
server_socket.listen(0)
print("listening")
client_socket, addr = server_socket.accept()
print("accepting")
data = client_socket.recv(65535)

print("receive : " + data.decode())

client_socket.send(data)
print("sned data")
client_socket.close()
print("종료")
```

## simple http server


```python
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.bind(('localhost', 80))
server_socket.listen(0)
print("listening")
client_socket, addr = server_socket.accept()
print("accepting")
data = client_socket.recv(65535)

print("receive : " + data.decode())
client_socket.close()
```

## simple http server2


```python
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.bind(('localhost', 80))
server_socket.listen(0)
print("listening")
client_socket, addr = server_socket.accept()
print("accepting")
data = client_socket.recv(65535)

print("receive : " + data.decode())

client_socket.send('HTTP/1.0 200 OK\r\n\r\n<font color=red>Hello</font>'.encode("utf-8"))

client_socket.close()
```

## if 함수를 통해 한번에 한번씩만 가능하게 


```python
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.bind(('localhost', 80))
server_socket.listen(0)
print("listening")

if True :
    client_socket, addr = server_socket.accept()
    print("accepting")
    data = client_socket.recv(65535)
    data = data.decode()
    
    headers = data.split("\r\n")
    print(headers[0])
    print(headers[0].split(" ")[1])
    
    client_socket.send('HTTP/1.0 200 OK\r\n\r\n<font color=red>Hello</font>'.encode("utf-8"))
    client_socket.close()
```

## datetime을 바탕으로 계속 돌아가는지 확인


```python
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.bind(('localhost', 80))
server_socket.listen(0)
print("listening")

while True :
    client_socket, addr = server_socket.accept()
    print("accepting")
    data = client_socket.recv(65535)
    print("receive : " + data.decode())
    
    header = 'HTTP/1.0 200 OK\r\n\r\n'    
    html = "hello" + str(datetime.now())
    client_socket.send(header.encode("utf-8"))
    client_socket.send(html.encode("utf-8"))
    client_socket.close()
```


```python
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.bind(('localhost', 80))
server_socket.listen(0)
print("listening")

while True :
    client_socket, addr = server_socket.accept()
    print("accepting")
    data = client_socket.recv(65535)
    print("receive : " + data.decode())
    
    header = 'HTTP/1.0 200 OK\r\n\r\n'    
    html = open()
    client_socket.send(header.encode("utf-8"))
    client_socket.send(html.encode("utf-8"))
    client_socket.close()
```

## html에서 불러오기


```python
filename = "/index2.html"

file = open("."+ filename, 'rt', encoding='utf-8')

print(file.read())
```


```python
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.bind(('localhost', 80))
server_socket.listen(0)
print("listening")

if  True :
    client_socket, addr = server_socket.accept()
    print("accepting")
    data = client_socket.recv(65535)    
    data = data.decode()
    
    headers = data.split("\r\n")
    filename = headers[0].split(" ")[1]
    
    
    file = open("."+ filename, 'rt', encoding='utf-8')
    html = file.read()
    
    header = 'HTTP/1.0 200 OK\r\n\r\n'    
    
    client_socket.send(header.encode("utf-8"))
    client_socket.send(html.encode("utf-8"))
    client_socket.close()
```


```python
import socket
import subprocess

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.bind(('localhost', 80))
server_socket.listen(0)
print("listening")

if  True :
    client_socket, addr = server_socket.accept()
    print("accepting")
    data = client_socket.recv(65535)    
    data = data.decode()
    #print(data)        
    
    try :    
        headers = data.split("\r\n")
        filename = headers[0].split(" ")[1]
        
        if '.py' in filename:
            header = 'HTTP/1.0 200 OK\r\n\r\n'   
            html = subprocess.check_output(['python.exe', '.' + filename])
            html = html.decode('cp949')
            client_socket.send(header.encode("utf-8"))    
            client_socket.send(html.encode("utf-8"))            
        elif '.html' in filename:
            file = open("."+ filename, 'rt', encoding='utf-8')
            html = file.read()    
            header = 'HTTP/1.0 200 OK\r\n\r\n'        
            client_socket.send(header.encode("utf-8"))
            client_socket.send(html.encode("utf-8"))
        elif '.jpg' in filename or '.ico' in filename or '.png' in filename:         
            client_socket.send('HTTP/1.1 200 OK\r\n'.encode())
            client_socket.send("Content-Type: image/jpg\r\n".encode())
            client_socket.send("Accept-Ranges: bytes\r\n\r\n".encode())
            file = open("." + filename, "rb")            
            client_socket.send(file.read())  
            file.close()               
        else :
            header = 'HTTP/1.0 404 File Not Found\r\n\r\n'        
            client_socket.send(header.encode("utf-8"))
    except Exception as e :
        print(e)         
    client_socket.close()
```


```python
import subprocess

output = subprocess.check_output(['python.exe', 'test.py'])
print(type(output))
print(output)
print(output.decode('cp949'))
```


```python
import socket
import threading
from datetime import datetime
import subprocess

def httpprocess(client_socket) :
    data = client_socket.recv(65535)   
    data = data.decode()
    print(data)
    try :    
        headers = data.split("\r\n")
        filename = headers[0].split(" ")[1]
        
        if '.py' in filename:
            
            html = subprocess.check_output(['python.exe', '.' + filename])
            html = html.decode('cp949')
            
            header = 'HTTP/1.0 200 OK\r\n'               
            client_socket.send(header.encode("utf-8"))
            client_socket.send("Content-Type: text/html\r\n\r\n".encode())                                    
            client_socket.send(html.encode("utf-8"))            
        elif '.html' in filename:
            file = open("."+ filename, 'rt', encoding='utf-8')
            html = file.read()    
            header = 'HTTP/1.0 200 OK\r\n'               
            client_socket.send(header.encode("utf-8"))
            client_socket.send("Content-Type: text/html\r\n\r\n".encode())            
            client_socket.send(html.encode("utf-8"))
        elif '.jpg' in filename or '.png' in filename:         
            client_socket.send('HTTP/1.1 200 OK\r\n'.encode())
            client_socket.send("Content-Type: image/jpg\r\n".encode())
            client_socket.send("Accept-Ranges: bytes\r\n\r\n".encode())
            file = open("." + filename, "rb")            
            client_socket.send(file.read())  
            file.close()               
        else :
            header = 'HTTP/1.0 404 File Not Found\r\n\r\n'        
            client_socket.send(header.encode("utf-8"))
    except Exception as e :
        print(e)         
    client_socket.close()


server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('localhost', 80))
server_socket.listen(0)
print("listening")

while  True :
    client_socket, addr = server_socket.accept()
    print("accepting")
    t = threading.Thread(target=httpprocess, args=(client_socket,))
    t.start()
```


```python
html = "hello @v1   test @v2   item @v3 "

html = html.replace("@v1", "안녕하세요")
html = html.replace("@v2", "이순신")
html = html.replace("@v3", "^^")


print(html)
```


```python
def render(html, data) :
    for v in data :
        html = html.replace("@"+v, data[v])
    return html

def renderfile(file, data) :
    html = open(file, "rt", encoding="utf-8").read()
    for v in data :
        html = html.replace("@"+v, data[v])
    return html

html = "hello @v1   test @v2   item @v3 "
data = {"v1":"안녕하세요",  "v2":"이순신", "v3":"^^"}
html = render(html, data)
print(html)


data = {"title":"나의 홈페이지",  "name":"이순신", "email":"^^"}
print(renderfile("template.html", data))
```


```python
from IPython.core.display import HTML

def render(html, data) :
    for v in data :
        html = html.replace("@"+v, data[v])
    return html

def renderfile(file, data) :
    html = open(file, "rt", encoding="utf-8").read()
    for v in data :
        html = html.replace("@"+v, data[v])
    return html


data = {"title":"나의 홈페이지",  "name":"이순신", "email":"^^"}
HTML(renderfile("template.html", data))
```



# 돌리는데 들어가는 py, html

## form.html

```html
<form action="http://127.0.0.1:80/" method=post>

    <input type=text name=id>  
    <input type=submit value="send">
    
</form>
```

## index0.html

```php+HTML
<html>
    <head>
        <meta charset="UTF-8">
    </head>
    <body>

<h1>Hello</h1>
한글 ㅏㅇㄹ마열ㅏ
    </body>

</html>
```

## index2.html

```html
<font color=red> My Web Server</font>
<br>
<img src="movie.jpg">
<br>
<img src="son7.jpg">
```

## test.py

````python
a = 10
b =20 
c = [1,2,3,4,454,545]

html ="""
<html>
<head>
<meta charset="utf-8">
</head>
<body>
<font color=red> 가나다라마바사아자차카타파하 </font>
<br>
<img src = 'son7.jpg'>
</body>
"""
print(html)
````

## test2.py

```python
def renderfile(file, data) :
    html = open(file, "rt", encoding="utf-8").read()
    for v in data :
        html = html.replace("@"+v, data[v])
    return html

data = {"title":"나의 홈페이지",  "name":"이순신", "email":"lee@gmail.com"}
print(renderfile("template.html", data))
```

## test3.py

```python
a=10
b=20

html = """




"""

print("hello")
a = 135
for i in str(a) :
   print(f"<img src={i}.png width=40 >")
```

## web.py

```python
import socket

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.bind(('localhost', 82))
server_socket.listen(3)
print("listening")

while  True :
    client_socket, addr = server_socket.accept()
    print("accepting")
    data = client_socket.recv(65535)    
    data = data.decode()
    print(data)        
    
    try :    
        headers = data.split("\r\n")
        filename = headers[0].split(" ")[1]

        if '.html' in filename:
            file = open("."+ filename, 'rt', encoding='utf-8')
            html = file.read()    
            header = 'HTTP/1.0 200 OK\r\n\r\n'        
            client_socket.send(header.encode("utf-8"))
            client_socket.send(html.encode("utf-8"))
        elif '.jpg' in filename or '.ico' in filename:         
            client_socket.send('HTTP/1.1 200 OK\r\n'.encode())
            client_socket.send("Content-Type: image/jpg\r\n".encode())
            client_socket.send("Accept-Ranges: bytes\r\n\r\n".encode())
            file = open("." + filename, "rb")            
            client_socket.send(file.read())  
            file.close()               
        else :
            header = 'HTTP/1.0 404 File Not Found\r\n\r\n'        
            client_socket.send(header.encode("utf-8"))
    except Exception as e :
        print(e)         
    client_socket.close()
```

## client.ipynb

```python
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

sock.connect(('localhost', 12345))
print('서버점속성공')

sock.send('hello'.encode())
print(('send message'))

data = sock.recv(65535)

print("receive:" + data.decode())
print("종료")
```

