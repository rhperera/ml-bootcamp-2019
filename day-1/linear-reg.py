# Linear Regression example with sklearn

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from numpy import arange

data = pd.read_csv("population-profit.txt")

#print(data.head())
plot = data.plot.scatter(x='population', y='profit')
#plt.show()

# Let's load our values into two matrices of 97x1
X = pd.DataFrame(data, columns=['population'])

# Let's add a bias term. Array is 1s
# X.insert(loc=0, column='bias', value=np.full((len(X), 1), 1.0))
# print(X.head())

y = pd.DataFrame(data, columns=['profit'])


'''
 We have 97 samples. Let's split these into train
 test sets. We will use 80% of samples for training
 and 20% of samples of testing
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Training input samples " + str(X_train.shape))
print("Testing input samples " + str(X_test.shape))
print("Training output samples " + str(y_train.shape))
print("Testing output samples " + str(y_test.shape))


lr_model = LinearRegression()

lr_model.fit(X_train, y_train, )
y_train_predict = lr_model.predict(X_train)

print(mean_squared_error(y_train, y_train_predict))
m = lr_model.coef_
c = lr_model.intercept_

line = m[0]*(arange(5,23)) + c[0]

plt.plot((arange(5,23)), line, 'r')
plt.show()
