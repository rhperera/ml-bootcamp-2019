# Logistic regression with sklearn

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("exam-scores.txt")

colors = ['red', 'blue']

plot = data.plot.scatter(x='exam1', y='exam2', c='admission', cmap=plt_colors.ListedColormap(colors))
plt.show()

X = pd.DataFrame(data, columns=['exam1', 'exam2'])

y = pd.DataFrame(data, columns=['admission'])

'''
 We have 101 samples. Let's split these into train
 test sets. We will use 80% of samples for training
 and 20% of samples of testing
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Training input samples " + str(X_train.shape))
print("Testing input samples " + str(X_test.shape))
print("Training output samples " + str(y_train.shape))
print("Testing output samples " + str(y_test.shape))

model = LogisticRegression()

model.fit(X_train, y_train)

print(model.predict([[100.0,100.0]]))