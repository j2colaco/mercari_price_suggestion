
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import gc


# In[2]:


from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


# code copied from https://www.kaggle.com/marknagelberg/rmsle-function
def rmsle(y_pred, y_test) : 
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))


# In[5]:


# Gets the count of most frequent words give a dataframe
def word_freq(df, col):
    word_frequency = {}
    word_frequency_lst = []
    for index,row in df.iterrows(): 
        for w in list(set(str(row[col]).split(' '))):
            if w not in word_frequency:
                word_frequency[w] = 1
            else:
                word_frequency[w] += 1

    for key, value in word_frequency.items():
        temp = [key, value]
        word_frequency_lst.append(temp)
    word_freq_df = pd.DataFrame(word_frequency_lst, columns=["unique_word", 'frequency'])
    word_freq_df = word_freq_df.sort_values(['frequency'], ascending=False)
    return word_freq_df


# # Read Data

# In[6]:


clean_data = pd.read_csv(
    '/Users/joashc/Downloads/mercari-price-suggestion-challenge/partially_clean_train_data.csv')
clean_data.shape


# ## Modeling Text
# - stemmed_item_description
#     - tdidf matrix
# - clean_brand_name
#     - One-hot-encode
# - clean_category_name
#     - One-hot-encode unique values if possible
# - clean_item_name
#     - tdidf matrix

# ### TF-IDF item_description

# In[7]:


item_description_df = clean_data['stemmed_item_description']
item_description_df.shape


# In[8]:


max_item_desc_features = 1500


# In[9]:


tfidf = TfidfVectorizer(max_features=max_item_desc_features)
x_tfidf = pd.DataFrame(tfidf.fit_transform(item_description_df).toarray())
x_tfidf.columns = ['item_desc_' + str(col) for col in x_tfidf.columns]
print(x_tfidf.shape)
x_tfidf.head(2)


# In[10]:


print(clean_data.shape)
clean_data_v2 = pd.concat([clean_data, x_tfidf], axis=1).drop(columns=['item_description', 
                                                                                         'stemmed_item_description'])
print(clean_data_v2.shape)


# In[11]:


# Delete dataframes from memory
del [[x_tfidf,item_description_df, clean_data]]
gc.collect()
clean_data = pd.DataFrame()
item_description_df=pd.DataFrame()
x_tfidf=pd.DataFrame()


# ### TF-IDF clean_item_name

# In[12]:


# item_name_df = clean_data_v2['clean_item_name']
# item_name_df.shape


# In[36]:


# max_item_name_features = 100


# In[37]:


# tfidf = TfidfVectorizer(max_features=max_item_name_features)
# item_name_tfidf = pd.DataFrame(tfidf.fit_transform(item_name_df).toarray())
# item_name_tfidf.columns = ['item_name_' + str(col) for col in item_name_tfidf.columns]
# print(item_name_tfidf.shape)
# item_name_tfidf.head(2)


# In[38]:


# %%time
# print(clean_data_v4.shape)
# clean_data_v5 = pd.concat([clean_data_v4, item_name_tfidf], axis=1).drop(columns=['clean_item_name', 
#                                                                                          'name'])
print(clean_data_v5.shape)


# In[39]:


# # Delete dataframes from memory
# del [[item_name_tfidf,item_name_df, clean_data_v4]]
# gc.collect()
# clean_data_v4 = pd.DataFrame()
# item_name_df=pd.DataFrame()
# item_name_tfidf=pd.DataFrame()


# ### Drop Unwanted Columns

# In[13]:


clean_data_v2 = clean_data_v2.drop(columns=list(clean_data_v2.select_dtypes(object)))
clean_data_v2.shape


# In[16]:


clean_data_v2 = clean_data_v2.drop(columns= ['item_condition_id','shipping'])
clean_data_v2.shape


# # Model Implementation

# In[17]:


X_train, X_test, y_train, y_test = train_test_split(clean_data_v2.drop(columns=['price']).reset_index(drop=True), 
                                                    clean_data_v2[['price']].reset_index(drop=True), 
                                                                  test_size=0.15, random_state=42)
print('Number of rows in train and validation data:', X_train.shape[0], y_train.shape)
print('Number of rows in test data:', X_test.shape[0], y_test.shape)


# In[18]:


X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)


# In[ ]:


# scaler = MinMaxScaler(feature_range=(0, 1))
# X_train = scaler.fit_transform(X_train)
# print('MinMaxScaler Complete')


# #### Model architecture parameters
# n_stocks = 500
# n_neurons_1 = 1024
# n_neurons_2 = 512
# n_neurons_3 = 256
# n_neurons_4 = 128
# n_target = 1

# In[19]:


X_train, X_val, y_train, y_val = train_test_split(X_train.reset_index(drop=True), 
                                                    y_train.reset_index(drop=True), 
                                                                  test_size=0.15, random_state=42)
print('Number of rows in train and validation data:', X_train.shape[0], y_train.shape)
print('Number of rows in validation data:', X_val.shape[0], y_val.shape)


# In[20]:


def step_decay(epoch):
    initial_lrate = 0.002
    drop = 0.5
    epochs_drop = 20
    lrate = initial_lrate * math.pow(drop,
    math.floor((1+epoch)/epochs_drop))
    return lrate


# In[21]:


num_epochs = 15
batch_size = 2213

all_val_predictions = pd.DataFrame()

train_val_rmsle = []

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=X_train.shape[1]))
model.add(BatchNormalization())
# model.add(Dropout(0.8))

model.add(Dense(264, activation='relu'))
model.add(BatchNormalization())
# model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
# model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
# model.add(Dropout(0.5))

model.add(Dense(1))
#     model.add(Activation('sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

lrate = LearningRateScheduler(step_decay)

model_hist = model.fit(X_train, y_train, validation_data=(X_val,y_val), 
                       batch_size=batch_size, epochs=num_epochs, verbose=1)


# In[ ]:


print(model_hist.history)


# In[ ]:


print('Train error is:', np.sqrt(mean_squared_error(y_train,model.predict(X_train))))

print('Validation error is:', np.sqrt(mean_squared_error(y_val,model.predict(X_val))))

print('Test error is:', np.sqrt(mean_squared_error(y_test,model.predict(scaler.transform(X_test)))))

