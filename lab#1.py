from hashlib import new
import padas as pd # type: ignore
import numpy as np # type: ignore
from sklearn import linear_model # type: ignore
x=np.array[(5,14,25,35,45,55)]#.reshape((-1,-1))
y=np.array[(5,20,14,32,22,38)]
model=linear_model.LinearRegression()
model.fit(x,y)
x-new=np.array([60]).reshape((-1,1))
y-new=model.predict(x-new)
print(y-new)
