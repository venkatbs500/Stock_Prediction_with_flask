#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf


# In[2]:


gs = yf.Ticker("GS")


# In[3]:


gs = gs.history(period = "max")


# In[4]:


gs


# In[5]:


gs.index


# In[6]:


gs.plot.line(y = "Close", use_index = True)


# In[7]:


del gs["Dividends"]
del gs["Stock Splits"]


# In[8]:


gs["Tomorrow"] = gs["Close"].shift(-1)


# In[9]:


gs


# In[10]:


gs["Target"] = (gs["Tomorrow"] > gs["Close"]).astype(int)


# In[11]:


gs


# In[12]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200, min_samples_split=100, random_state=1)
train = gs.iloc[:-100]
test = gs.iloc[-100:]
predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])


# In[13]:


from sklearn.metrics import precision_score 
preds = model.predict(test[predictors])


# In[14]:


preds


# In[15]:


import pandas as pd
preds = pd.Series(preds, index = test.index)


# In[16]:


preds


# In[17]:


precision_score(test["Target"],preds)


# In[18]:


combined = pd.concat([test["Target"],preds],axis = 1)


# In[19]:


combined.plot()


# In[22]:


gs


# In[23]:


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# In[24]:


def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)


# In[25]:


predictions = backtest(gs, model, predictors)


# In[26]:


predictions["Predictions"].value_counts()


# In[27]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[28]:


predictions["Target"].value_counts() / predictions.shape[0]


# In[29]:


horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = gs.rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"
    gs[ratio_column] = gs["Close"] / rolling_averages["Close"]
    
    trend_column = f"Trend_{horizon}"
    gs[trend_column] = gs.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors+= [ratio_column, trend_column]


# In[30]:


gs


# In[31]:


gs = gs.dropna()


# In[32]:


gs


# In[33]:


model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)


# In[34]:


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >=.6] = 1
    preds[preds <.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# In[35]:


predictions = backtest(gs, model, new_predictors)


# In[36]:


predictions["Predictions"].value_counts()


# In[37]:


precision_score(predictions["Target"], predictions["Predictions"])


