# Data Manupulation
import numpy as np
import pandas as pd

# Techinical Indicators
import talib as ta

# Plotting graphs
import matplotlib.pyplot as plt

# Machine learning
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

# Data fetching
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override()



df = pdr.get_data_yahoo('^NSEI', '2000-01-01', '2018-01-01')
df = df.dropna()
df = df.iloc[:,:4]
df.head()


df['S_10'] = df['Close'].rolling(window=10).mean()
df['Corr'] = df['Close'].rolling(window=10).corr(df['S_10'])
df['RSI'] = ta.RSI(np.array(df['Close']), timeperiod =10)
df['Open-Close'] = df['Open'] - df['Close'].shift(1)
df['Open-Open'] = df['Open'] - df['Open'].shift(1)
df = df.dropna()
X = df.iloc[:,:9]

y = np.where (df['Close'].shift(-1) > df['Close'],1,-1)

split = int(0.7*len(df))

X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

model = LogisticRegression()

model = model.fit (X_train,y_train)

pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))

probability = model.predict_proba(X_test)

print probability

predicted = model.predict(X_test)
print metrics.confusion_matrix(y_test, predicted)

print metrics.classification_report(y_test, predicted)

print model.score(X_test,y_test)

cross_val = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)

print cross_val

print cross_val.mean()
