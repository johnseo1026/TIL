# 챗봇(Dialogflow)

> dialogflow 구글 아이디 로그인

![image-20200204135515083](images/image-20200204135515083.png)

> create intent 클릭

## orderfood

> orderfood 제목 설정

![image-20200204135605259](images/image-20200204135605259.png)

![image-20200204144222118](images/image-20200204144222118.png)

![image-20200204135625186](images/image-20200204135625186.png)

> 만든후 상단의 save  클릭

![image-20200204144242328](images/image-20200204144242328.png)\



### orderfood_menu



> orderfood에 마우스 커서를 올리면 Add Follow-up intent나오는데 클릭

![image-20200204140014819](images/image-20200204140014819.png)

> orderfood_menu로 지정

![image-20200204140201914](images/image-20200204140201914.png)

![image-20200204140222127](images/image-20200204140222127.png)

![image-20200204141556436](images/image-20200204141556436.png)

> 체크하면 리스트로 나옴

![image-20200204141937247](images/image-20200204141937247.png)

![image-20200204140236573](images/image-20200204140236573.png)



![image-20200204144406148](images/image-20200204144406148.png)



#### orderfood_menu_confirm



> 위와 동일하게 menu에 커서 올려서 Add Follow-up intent 나오는데 클릭

![image-20200204140302353](images/image-20200204140302353.png)

![image-20200204140400134](images/image-20200204140400134.png)

![image-20200204140436573](images/image-20200204140436573.png)



> 여기서는 $food가 안되기 때문에 앞에서 짜장면에 대한 것을 가져와야함 





![image-20200204143933004](images/image-20200204143933004.png)

![image-20200205162518448](images/image-20200205162518448.png)



> 역시 앞에서 가져와야하기에 추가

![image-20200204144031246](images/image-20200204144031246.png)

![image-20200204144442627](images/image-20200204144442627.png)



## order2



> intent order2로 생성후

![image-20200204152650300](images/image-20200204152650300.png)

> create 하기

![image-20200204152717582](images/image-20200204152717582.png)

> 생성 
>
> ex)
>
> 위에것은 짜장 - 3, 짬뽕 - 5 형식
>
> 아래것은 짜장

![image-20200204152834300](images/image-20200204152834300.png)

![image-20200204152845350](images/image-20200204152845350.png)

![image-20200204152857899](images/image-20200204152857899.png)

## weather

![image-20200205162604535](images/image-20200205162604535.png)

![image-20200205162614983](images/image-20200205162614983.png)







## query





![image-20200205141134283](images/image-20200205141134283.png)

![image-20200205162420153](images/image-20200205162420153.png)





# 코드

> pip install gtts

```python
import requests
import urllib
from bs4 import BeautifulSoup
import json
from gtts import gTTS
import IPython.display as ipd
from IPython.core.display import HTML
```

```python
# Dialogflow 불러오는 함수
def get_answer(text, sessionId):
    data_send = {
        'query': text, 'sessionId': sessionId,
        'lang': 'ko', 'timezone' : 'Asia/Seoul'
    }
    data_header = {
        'Authorization': 'Bearer 862fdf19603c4e6186b010e7348979a8',
        'Content-Type': 'application/json; charset=utf-8'
    }

    dialogflow_url = 'https://api.dialogflow.com/v1/query?v=20150910'
    res = requests.post(dialogflow_url, data=json.dumps(data_send), headers=data_header)
    if res.status_code == requests.codes.ok:
       return res.json()    
    return {}
```

```python
# 날씨 함수
def getWeather(city) :    
    url = "https://search.naver.com/search.naver?query="
    url = url + urllib.parse.quote_plus(city + "날씨")
    print(url)
    bs = BeautifulSoup(urllib.request.urlopen(url).read(), "html.parser")
    temp = bs.select('span.todaytemp')    
    desc = bs.select('p.cast_txt')    
    return  {"temp":temp[0].text, "desc":desc[0].text}    
    #return temp[0].text + "˚/" + desc[0].text


#print(getWeather("제주도")["desc"])
```

```python
# 단어찾기 함수
def getQuery(word) :
    url = "https://search.naver.com/search.naver?where=kdic&query="
    url = url + urllib.parse.quote_plus(word)
    print(url)
    bs = BeautifulSoup(urllib.request.urlopen(url).read(), "html.parser")
    output = bs.select('p.txt_box')
    
    return [node.text for node in output]
    #return output[0].text

#print(getQuery("갑분싸")[0])
#print(getQuery("python")[1])
```

```python
# 음성변환 함수
def printtts(text):
    tts = gTTS(text=text, lang='ko')
    tts.save("output.mp3")
    print(text)
    ipd.display(ipd.Audio('output.mp3', autoplay = True))
    
#print(printtts("나는 배고프다"))
```

```python
while True :
    txt = input("->")
    dict = get_answer(txt, 'user01')
    answer = dict['result']['fulfillment']['speech'] 
    intentName =dict['result']['metadata']['intentName']

    if intentName == 'query' :
        word = dict['result']['parameters']['any']
        text = getQuery(word)[0]
        printtts(text)
        
    elif intentName == 'order2' :
        price = {"짜장면":5000, "짬뽕":10000, "탕수육":20000}
        params = dict['result']['parameters']['food_number']
        
        output = [ food.get("number-integer", 1)*price[food["food"]]  for food in params ] 
        printtts(str(sum(output)))  # 문자열로 직접 바꿔줘야한다 printtts에서는
        
    elif dict['result']['metadata']['intentName'] == 'weather' and  dict['result']['actionIncomplete'] == False :        
        date = dict['result']['parameters']['date']
        geo_city = dict['result']['parameters']['geo-city']                    
            
        info = getWeather(geo_city)            
        printtts(f"{geo_city} 날씨 정보 : {info['temp']} /  {info['desc']}")
        #print("날씨 ", date, geo_city)       
    else :
        printtts("Bot:" + answer)   
```









# 서버 만들기

new로 들어가서 text

```python
from flask import Flask, request, jsonify
import requests
import urllib
import json
from bs4 import BeautifulSoup

def getWeather(city) :    
    url = "https://search.naver.com/search.naver?query="
    url = url + urllib.parse.quote_plus(city + "날씨")
    print(url)
    bs = BeautifulSoup(urllib.request.urlopen(url).read(), "html.parser")
    temp = bs.select('span.todaytemp')    
    desc = bs.select('p.cast_txt')    
    return  {"temp":temp[0].text, "desc":desc[0].text}    


app = Flask(__name__)

@app.route('/')
def home():   # home 말고 마음대로 해도 상관없음
    name = request.args.get("name")
    item = request.args.get("item")
    return "hellooo~~~ hi" + name + item

@app.route('/abc')
def home2():   # 같게 만들필요는 없음
    return "test!!!"

@app.route('/weather')
def weather():
    city = request.args.get("city")
    info = getWeather(city)
    return jsonify(info)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)   # debug=True를 해야 바로바로 수정이 가능
```

> server.py로 저장
>
> 아나콘다 프롬프트 들어가서 서버 만든 폴더로 이동
>
> python server.py 를 치면됨





ip 찾는법

cmd에서 ipconfig









# ngrok

![image-20200205174639696](images/image-20200205174639696.png)

> 노란 표시된것을 url에 복사
>
> dialogflow에 fulfillment로

![image-20200205174821913](images/image-20200205174821913.png)

![image-20200205174919851](images/image-20200205174919851.png)

> 밑에 fulfillment 변경하면 
>
> 서버에 있는 것으로 결과가 표시됨
>
> 아래에 있는 Heloo~~
>
> 만약 서버가 죽으면 원래 responses에 있는것으로 표현됨

![image-20200205175423355](images/image-20200205175423355.png)



![image-20200205175024395](images/image-20200205175024395.png)





