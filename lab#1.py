import sklearn
import padas as pd
import numpy as np
from sklearn import linear_model
x=np.array{(5,14,25,35,45,55)}.reshape((-1,-1))
y=bp.array{(5,20,14,32,22,38)}
model=linear_model.linear regression()
model fit(x,y)
x-new=np.array([60]).reshape((-1,1))
y-new=model.predict(x-new)
print(y-new)
