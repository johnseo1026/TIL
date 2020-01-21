

> visual studio code 다운받아서 실행\
>
> 아래를 dev 폴더를 만든후 안에  main.py로 저장
>
> 그후 아나콘다프롬프트로 들어가서  set FLASK_APP=main.py
>
> flask run 코드 실행
>
> 실행후 나온주소에 Hello, World! 뜨는지 확인

```python
from flask import Flask, escape, request

app = Flask(__name__)

# set FLASK_APP=main.py
# flask run

@app.route('/')
def hello():
    name = request.args.get("name", "World")
    return f'Hello, {escape(name)}!'

@app.route('/hi', methods=['POST'])
def hi():
    return{
    "version": "2.0",
    "template": {
        "outputs": [
            {
                "simpleText": {
                    "text": "간단한 텍스트 요소입니다."
                }
            }
        ]
    }
}
```

> postman 다운로드

```python
from flask import Flask, escape, request


app = Flask(__name__)

# set FLASK_APP=main.py
# flask run
db = {}
id = 0

@app.route('/users', methods=['POST'])
def create_user():
    global id
    print(request)
    body = request.json
    print(body)
    body['id'] = id
    # todo body에  id를 넣어준다.
    db[str(id)] = body
    id+=1
    return body

@app.route('/users/<id>}', methods=['GET'])
def select_user():
    return db[id]
    
def delete_user():
    pass

def update_user():
    pass

@app.route('/hi', methods=['POST'])
def hi():
    return{
    "version": "2.0",
    "template": {
        "outputs": [
            {
                "simpleText": {
                    "text": "간단한 텍스트 요소입니다."
                }
            }
        ]
    }
}
```

> postman에서 untitled Request 아래를 post로 변경 http://127.0.0.1:5000/users
>
> body에  raw 누르고 오른쪽 json으로 변경
>
> ``` json
> {
> 	"name" : "gugu",
> 	"phone" : "010-1234-5678"
> }
> ```
>
> 
>
> 하고 send 누르면 id 하나씩 올라가는거 확인



