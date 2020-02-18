from sklearn.model_selection import train_test_split
import numpy as np
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4],[5,6],[7,9],[4,9],[23,1],[4,7],[3,8],[1, 2], [3, 4], [1, 2], [3, 4],[5,6],[7,9],[4,9],[23,1],[4,7],[3,8]])
y = np.array([0, 1, 1, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0])
# print(len(y))
a,b = y.shape,X.shape
for i in b:
    print(i)
# xtrain,xtest,ytrain,ytest=train_test_split(X,y,stratify=y,test_size=3)
# print(xtrain,'***')
# print(xtest,'&&&')
# print(ytrain,'^^^^')
# print(ytest,'$$$$')