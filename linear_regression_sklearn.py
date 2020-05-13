import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import random as rand
import quandl

plt.style.use('ggplot')
rand.seed(11)


def create_dataset(n, variance, step=2, corr=False):

    val = 1
    y = []
    for i in range(n):
        num = val + rand.randrange(-variance, variance)
        y.append(num)
        if corr and corr == 'pos':
            val += step
        elif corr and corr == 'neg':
            val -= step
    x = [i for i in range(len(y))]
    
    return np.array(x, dtype='float64'), np.array(y, dtype='float64')


def data_from_quandl(ticker):

    key = "ADD KEY HERE"
    data = quandl.get(ticker, api_key=key)
    returns = data['Last'].pct_change(periods=1)[-1000:]

    return returns


def average(nums):
    
    count = 0
    total = 0
    for i in nums:
        count += 1
        total += i
    return total / count


def x_y_bars():

    x_bar = average(x)
    y_bar = average(y)
    xy_bar = average(x*y)
    x2_bar = average(x**2)

    return x_bar, y_bar, xy_bar, x2_bar

    

x, y = create_dataset(200, 50, 2, 'pos')
x, y = x.reshape(1, -1), y.reshape(1, -1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)
conf = clf.score(X_test, y_test)

print(conf)

##new_x = np.array([2.5, 5.5, 8.5])
##predict_y = (m * new_x) + b

##
##plt.scatter(x, y, color='blue')
##plt.plot(x, reg_line, color='red')
##plt.legend(['R squared: '+str(r_squared())])
##plt.show()
##
##
##
