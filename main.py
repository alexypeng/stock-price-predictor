import yfinance as yf
import pandas as pd
from datetime import date
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Getting data from yfinance
ticker = open("stock-ticker.txt", "r").read()
data = yf.Ticker(ticker)

# Setting up data frame containing the data
data = data.history(period="max")
data.index = pd.to_datetime(data.index).date

del data['Dividends']
del data['Stock Splits']

# Adding columns to the data frame
data['Tomorrow'] = data['Close'].shift(-1)
data['Target'] = (data['Tomorrow'] > data['Close']).astype(int)

estimators = 100
min_samples = 100

model = RandomForestClassifier(n_estimators=estimators, min_samples_split=min_samples, random_state=1)

train = data.iloc[:-100]
test = data.iloc[-100:]

predictors = ['Close', 'Volume', 'Open', 'High', 'Low']
model.fit(train[predictors], train['Target'])

preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train['Target'])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index)
    return pd.concat([test[predictors], preds], axis=1)

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    



