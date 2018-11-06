from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
#import tensorflow as tf  [commented out because havent used yet]
from sklearn.preprocessing import MinMaxScaler

#LSTM neural network prediction model which gets data from alpha vantage

#API key, need to make private if used too much
api_key = '6DMWN75E9N3H4RLZ'

# Ticker
ticker = "AAPL" 

# get JSON file with stock data
# example: https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&outputsize=full&apikey=6DMWN75E9N3H4RLZ
url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)

#CSV file name to save to
file_to_save = 'stock_market_data-%s.csv'%ticker

# If you haven't already saved data,
# Pandas dataframe <- Date, Low, High, Close, Open, and Index
if not os.path.exists(file_to_save):
    with urllib.request.urlopen(url_string) as url:
        data = json.loads(url.read().decode())
        data = data['Time Series (Daily)']
        df = pd.DataFrame(columns=['Date','Low','High','Close','Open', 'Adjusted Close'])
        for k,v in data.items():
            date = dt.datetime.strptime(k, '%Y-%m-%d')
            #we also want adjusted close because stocks split over time
            data_row = [date.date(),float(v['3. low']),float(v['2. high']),
                        float(v['4. close']),float(v['1. open']), float(v['5. adjusted close'])]
            df.loc[-1,:] = data_row
            df.index = df.index + 1
        print(df)
    print('Data saved to : %s'%file_to_save)        
    df.to_csv(file_to_save)
#if CSV for stock was already created, then load it
else:
    print('File already exists. Loading data from CSV')
    df = pd.read_csv(file_to_save)
        
df = df.sort_values('Date')
df.head()

print(df.head())


adjusted_price  = df['Adjusted Close']

plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]), adjusted_price)
plt.title(ticker, fontsize=18)
plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.show()

# First calculate the mid prices from the highest and lowest 
high_prices = df.loc[:,'High'].values
low_prices = df.loc[:,'Low'].values
#mid_prices = (high_prices+low_prices)/2.0
mid_prices = df.loc[:,'Adjusted Close'].values

#get number of data points or index of most recent data point
MR_data_point = df.index[1]
print(MR_data_point)

train_data = mid_prices[:MR_data_point]
test_data = mid_prices[MR_data_point:]

# for 2D array support
scaler = MinMaxScaler()
train_data = train_data.reshape(-1, 1)
test_data = test_data.reshape(-1, 1)

smoothing_window_size = 5000
for di in range(0,10000,smoothing_window_size):
    try:
        fitData = train_data[di:di+smoothing_window_size]
        print(len(fitData))
        print('iterated')
        scaler.fit(fitData)
        train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])
    except ValueError:
        break

try:
    scaler.fit(train_data[di+smoothing_window_size:,:])
    train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])
    scaler.fit(train_data[di+smoothing_window_size:,:])
    train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])
except ValueError:
    print("Array with 0 sample(s) for MinMaxScaler")
    pass

# Reshape both train and test data
train_data = train_data.reshape(-1)

# Normalize test data
test_data = scaler.transform(test_data).reshape(-1)

# Now perform exponential moving average smoothing
# So the data will have a smoother curve than the original ragged data
EMA = 0.0
gamma = 0.1
for ti in range(11000):
  EMA = gamma*train_data[ti] + (1-gamma)*EMA
  train_data[ti] = EMA

# Used for visualization and test purposes
all_mid_data = np.concatenate([train_data,test_data],axis=0)


window_size = 100
N = train_data.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pred_idx in range(window_size,N):
    
    if pred_idx >= N:
        date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
    else:
        date = df.loc[pred_idx,'Date']
        
    std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
    mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
    std_avg_x.append(date)

print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))

