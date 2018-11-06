import quandl
import pandas as pd
import numpy as np
import datetime
import api_file as api
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection, svm

#Linear regression, with data from quandl (moving to Alpha Vantage later)

apiKey = api.getApiKey()

endDate = '2017-12-19'

#next is getting user input and making more dynamic
user_input = "AMZN"
stock_to_analyze = "WIKI/" + user_input

df = quandl.get(stock_to_analyze, end_date=endDate, api_key=apiKey)
df = df[['Adj. Close']]

forecast_out = int(30)

real_data = quandl.get(stock_to_analyze, start_date=endDate, end_date='2018-1-27')
real_data = real_data[['Adj. Close']]
real_data = real_data['Adj. Close'].values.tolist()


df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)

X = np.array(df.drop(['Prediction'],1))
X = preprocessing.scale(X)

X_forecast = X[-forecast_out:]
X = X[:-forecast_out]

y = np.array(df['Prediction'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

clf = LinearRegression()
clf.fit(X_train,y_train)

confidence = clf.score(X_test, y_test)
print("confidence: ", confidence)

forecast_prediction = clf.predict(X_forecast)
print("predicted_data: ")
print(forecast_prediction)
print("actual_data: ")
print(real_data)

plt.plot(forecast_prediction, label='Prediction line')
plt.plot(real_data, label = 'Actual price line')
plt.legend()
plt.axis('auto')
plt.title(user_input + " Stock Prediction")
plt.xlabel('Days After ' + endDate)
plt.ylabel('Price ($)')
plt.show()