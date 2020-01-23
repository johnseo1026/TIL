# XOR


```python
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.optimizers import Adam
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from keras.callbacks import LambdaCallback
from keras.callbacks import EarlyStopping
from keras.layers import LeakyReLU

def lambdaf_(epoch, logs, step) : 
    if epoch % step == 0 : print(f"{epoch} => {logs}")                
        
def printepoch(step) :   
    return LambdaCallback(on_epoch_end=lambda epoch, logs: lambdaf_(epoch, logs, step)  )

```


```python
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])
```


```python
model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Activation('sigmoid'))

model.add(Dense(10))
model.add(Activation('sigmoid'))

model.add(Dense(10))
model.add(Activation('sigmoid'))
          
model.add(Dense(10))
model.add(Activation('sigmoid'))

model.add(Dense(10))
model.add(Activation('sigmoid'))

model.add(Dense(10))
model.add(Activation('sigmoid'))

model.add(Dense(5))
model.add(Activation('sigmoid'))

model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()                    # 층이 깊어졌을때 위로갈수록 학습이 덜된다
```

![image-20200123102544543](images/image-20200123102544543.png)

```python
model.compile(loss='binary_crossentropy', optimizer='adam')
```


```python
model.fit(X, y, batch_size=4, epochs=6000, verbose=0)
print(model.predict_proba(X))   # 행렬은 4x1이다. sample 수 x 클래스수
```

![image-20200123102609537](images/image-20200123102609537.png)

```python
p = model.predict_proba(X)
print((p>0.5)*1)
```

![image-20200123102619217](images/image-20200123102619217.png)

```python
xx, yy = np.mgrid[0:1:0.02, 0:1:0.02]
grid = np.c_[xx.flatten(), yy.flatten()]
print(grid.shape)
h = model.predict_proba(grid)

print(type(h))
print(h.shape)

colors = ["red"  if i>  0.5  else  "blue"  for i  in h  ]
plt.scatter(xx.flatten(), yy.flatten(), color = colors, alpha=0.2)  
plt.savefig('xor.png')
```

![image-20200123102643626](images/image-20200123102643626.png)

```python
from sklearn.datasets import make_moons
x_data, y_data = make_moons(n_samples=500, noise=0.1)   #  x 는 2차원   y는 레이블    0   1

colors = ["red"  if i == 1  else   "blue"    for i  in y_data  ]

plt.scatter(x_data[:,0], x_data[:,1], color = colors, alpha=1.0)  
print(y_data.shape)
```

![image-20200123102700442](images/image-20200123102700442.png)

```python
model = Sequential([
    Dense(5, activation='sigmoid',input_dim=2),   
    Dense(5, activation='sigmoid'),
    Dense(5, activation='sigmoid'),
    Dense(5, activation='sigmoid'),
    Dense(10, activation='sigmoid'),
    Dense(10, activation='sigmoid'),
    Dense(10, activation='sigmoid'),
    Dense(10, activation='sigmoid'),
    Dense(10, activation='sigmoid'),
    Dense(10, activation='sigmoid'),
    Dense(1, activation='sigmoid'),
])
model.compile(loss='binary_crossentropy', optimizer=Adam(0.01))  
model.fit(x_data, y_data, batch_size=100, epochs=3000, verbose=0,
          validation_data=(x_data, y_data))   # 변경되는 곳

colors = ["red"  if i == 1  else   "blue"    for i  in y_data  ]
plt.scatter(x_data[:,0], x_data[:,1], color = colors, alpha=1.0)
xx, yy = np.mgrid[-1.5:2.5:0.05, -1:1.5:0.05]    # 변경되는곳
grid = np.c_[xx.ravel(), yy.ravel()]
h = model.predict_proba(grid)
colors = ["red"  if i>  0.5  else  "blue"  for i  in h  ]
plt.scatter(xx.flatten(), yy.flatten(), color = colors, alpha=0.2)  
```

![image-20200123102718914](images/image-20200123102718914.png)

```python
model = Sequential([
    Dense(5, activation='sigmoid',input_dim=2),   
    Dense(5, activation='sigmoid'),
    Dense(10, activation='sigmoid'),
    Dense(10, activation='sigmoid'),
    Dense(10, activation='sigmoid'),
    Dense(10, activation='sigmoid'),
    Dense(10, activation='sigmoid'),
    Dense(10, activation='sigmoid'),
    Dense(10, activation='sigmoid'),
    Dense(10, activation='sigmoid'),
    Dense(1, activation='sigmoid'),
])
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
model.fit(x_data, y_data, batch_size=100, epochs=3000, verbose=0,
          validation_data=(x_data, y_data),
          callbacks=[printepoch(500)])
```

![image-20200123102736330](images/image-20200123102736330.png)


```python
colors = ["red"  if i == 1  else   "blue"    for i  in y_data  ]
plt.scatter(x_data[:,0], x_data[:,1], color = colors, alpha=1.0)
xx, yy = np.mgrid[-1.5:2.5:0.05, -1:1.5:0.05]    # 변경되는곳
grid = np.c_[xx.ravel(), yy.ravel()]
h = model.predict_proba(grid)
colors = ["red"  if i>  0.5  else  "blue"  for i  in h  ]
plt.scatter(xx.flatten(), yy.flatten(), color = colors, alpha=0.2)  
```

![image-20200123102748274](images/image-20200123102748274.png)


# 깊은 네트워크 학습


```python
model = Sequential([
    Dense(5, activation = LeakyReLU(alpha=0.1),input_dim=2),
    Dense(10, activation = LeakyReLU(alpha=0.1)),
    Dense(10, activation = LeakyReLU(alpha=0.1)),
    Dense(10, activation = LeakyReLU(alpha=0.1)),
    Dense(10, activation = LeakyReLU(alpha=0.1)),
    Dense(10, activation = LeakyReLU(alpha=0.1)),
    Dense(10, activation = LeakyReLU(alpha=0.1)),
    Dense(10, activation = LeakyReLU(alpha=0.1)),    
    Dense(10, activation = LeakyReLU(alpha=0.1)),
    Dense(10, activation = LeakyReLU(alpha=0.1)),
    Dense(10, activation = LeakyReLU(alpha=0.1)),
    Dense(1, activation='sigmoid'),
          ])
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
```


```python
model.fit(x_data, y_data, batch_size=100, epochs=3000, verbose=0,
          validation_data=(x_data, y_data),
          callbacks=[printepoch(500)])
```

![image-20200123102800828](images/image-20200123102800828.png)


```python
def createModel(layers, activation, input_dim) :    
    model = Sequential()        
    d = layers.pop(0)
    model.add(Dense(d, activation=activation,input_dim =input_dim))
    for l in layers :
        model.add(Dense(l, activation=activation))
    model.add(Dense(1, activation='sigmoid'))    
    return model

    
model = createModel([5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5], LeakyReLU(alpha=0.1), 2) 
```

