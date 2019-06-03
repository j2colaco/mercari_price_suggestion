import pandas as pd
import numpy as np
import re
import gc

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

import matplotlib.pyplot as plt


# In[4]:


# code copied from https://www.kaggle.com/marknagelberg/rmsle-function
def rmsle(y_pred, y_test) : 
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))


print('Reading data...')
clean_data = pd.read_csv(
    '~/partially_clean_train_data.csv')
print(clean_data.shape)

clean_data = clean_data.sample(200000)


# ### One-hot-encoding Brand Name

# In[16]:
unique_brand_names = pd.DataFrame(clean_data.clean_brand_name.value_counts())
min_brand_freq = 500
print('There are', unique_brand_names[unique_brand_names['clean_brand_name']>=min_brand_freq].shape[0],
      'brand names that occur >=',min_brand_freq, 'times in the dataset.','Will one-hot-encode these brands only.')


# In[17]:

clean_brand_name_df = clean_data['clean_brand_name']
print('Brand name shape:', clean_brand_name_df.shape)


# In[18]:


clean_brand_name_df = pd.get_dummies(clean_brand_name_df)
print('Shape of one-hot-brand name:', clean_brand_name_df.shape)


# In[19]:


drop_brand_col_lst = []
keep_brand_col_lst = list(unique_brand_names[unique_brand_names['clean_brand_name']>=min_brand_freq].index.values)

for col_name in clean_brand_name_df.columns:
    if col_name not in keep_brand_col_lst:
        drop_brand_col_lst.append(col_name)

clean_brand_name_df_v2 =clean_brand_name_df.drop(columns=drop_brand_col_lst)
clean_brand_name_df_v2.columns = ['brand_' + str(col) for col in clean_brand_name_df_v2.columns]
print('Shape of brand name after taking out cols:', clean_brand_name_df_v2.shape)


# In[20]:


clean_data_v2 = pd.concat([clean_data.reset_index(drop=True)
                                   , clean_brand_name_df_v2.reset_index(drop=True)],
                                  axis=1).drop(columns=['brand_name','clean_brand_name'])
print('Shape of clean_data:', clean_data_v2.shape)


# Delete the variables from memory

# In[21]:


del [[clean_brand_name_df,clean_brand_name_df_v2, clean_data]]
gc.collect()
clean_data = pd.DataFrame()
clean_brand_name_df=pd.DataFrame()
clean_brand_name_df_v2=pd.DataFrame()


# ### One-hot-encoding Assigned Category and Assigned Sub Category

# In[22]:


clean_category_name_df = clean_data_v2[['assigned_category', 'assigned_sub_category']]
print('Shape of category', clean_category_name_df.shape)


clean_category_name_df_v2 = pd.get_dummies(clean_category_name_df)
print('Shape of category after OHE:',clean_category_name_df_v2.shape)


# In[26]:


# concat with main dataset on clean_category_name
print('Before shape:',clean_data_v2.shape)
clean_data_v3 = pd.concat([clean_data_v2, clean_category_name_df_v2], axis=1)
print('Clean data after shape:', clean_data_v3.shape)


# In[29]:


# Delete dataframes from memory
del [[clean_category_name_df,clean_category_name_df_v2, clean_data_v2]]
gc.collect()
clean_data_v2 = pd.DataFrame()
clean_category_name_df=pd.DataFrame()
clean_category_name_df_v2=pd.DataFrame()


# ### TF-IDF item_description

# In[30]:


item_description_df = clean_data_v3['stemmed_item_description']
print('Item desc shape:',item_description_df.shape)


# In[31]:


max_item_desc_features = 1500


# In[32]:


tfidf = TfidfVectorizer(max_features=max_item_desc_features)
x_tfidf = pd.DataFrame(tfidf.fit_transform(item_description_df).toarray())
x_tfidf.columns = ['item_desc_' + str(col) for col in x_tfidf.columns]
print('TDIDF shape:', x_tfidf.shape)

print('Before shape:', clean_data_v3.shape)
clean_data_v4 = pd.concat([clean_data_v3, x_tfidf], axis=1).drop(columns=['item_description', 
                                                                                         'stemmed_item_description'])
print('After shape:', clean_data_v4.shape)

# Delete dataframes from memory
del [[x_tfidf,item_description_df, clean_data_v3]]
gc.collect()
clean_data_v3 = pd.DataFrame()
item_description_df=pd.DataFrame()
x_tfidf=pd.DataFrame()


# ### TF-IDF clean_item_name

# In[35]:


item_name_df = clean_data_v4['clean_item_name']
print('Item name shape:', item_name_df.shape)


# In[36]:


max_item_name_features = 100


# In[37]:


tfidf = TfidfVectorizer(max_features=max_item_name_features)
item_name_tfidf = pd.DataFrame(tfidf.fit_transform(item_name_df).toarray())
item_name_tfidf.columns = ['item_name_' + str(col) for col in item_name_tfidf.columns]
print('TDIDF item name shape:', item_name_tfidf.shape)

print(clean_data_v4.shape)
clean_data_v5 = pd.concat([clean_data_v4, item_name_tfidf], axis=1).drop(columns=['clean_item_name', 
                                                                                         'name'])
print(clean_data_v5.shape)

# Delete dataframes from memory
del [[item_name_tfidf,item_name_df, clean_data_v4]]
gc.collect()
clean_data_v4 = pd.DataFrame()
item_name_df=pd.DataFrame()
item_name_tfidf=pd.DataFrame()


# ### One hot encode item condition

# In[40]:


item_condition_df = clean_data_v5[['item_condition_id']]
print('Item condition shape',item_condition_df.shape)


# In[41]:


item_condition_df['item_condition_id'] = item_condition_df['item_condition_id'].astype(object)


# In[42]:


item_condition_df_v2 = pd.get_dummies(item_condition_df)
print('Item condition shape:', item_condition_df_v2.shape)

print('Before shape:', clean_data_v5.shape)
clean_data_v6 = pd.concat([clean_data_v5, item_condition_df_v2], axis=1)
print('After shape:', clean_data_v6.shape)

# Delete dataframes from memory
del [[item_condition_df,item_condition_df_v2, clean_data_v5]]
gc.collect()
clean_data_v5 = pd.DataFrame()
item_condition_df_v2=pd.DataFrame()
item_condition_df=pd.DataFrame()


# ### Shipping
# - Change value of 0 to -1

# In[45]:


clean_data_v6['shipping'] = clean_data_v6['shipping'].replace([0], [-1])


# ### Drop Unwanted Columns



clean_data_v6 = clean_data_v6.drop(columns=[ 'item_condition_id', 
                                                            'category_name', 'clean_category_name', 
                                                            'assigned_category',
                                                           'assigned_sub_category'])
print('Shape after adding shipping:', clean_data_v6.shape)




X_train, X_test, y_train, y_test = train_test_split(clean_data_v6.drop(columns=['price']).reset_index(drop=True), 
                                                    clean_data_v6[['price']].reset_index(drop=True), 
                                                                  test_size=0.2, random_state=42)
print('Number of rows in train and validation data:', X_train.shape[0], y_train.shape)
print('Number of rows in test data:', X_test.shape[0], y_test.shape)

# del variable from memory
del [[clean_data_v6]]
gc.collect()
clean_data_v6 = pd.DataFrame()

scaler = MinMaxScaler(feature_range=(0, 1))
columns = X_train.columns
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=columns)
print('MinMaxScaler Complete')

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)



X_train, X_val, y_train, y_val = train_test_split(X_train.reset_index(drop=True), 
                                                    y_train.reset_index(drop=True), 
                                                                  test_size=0.2, random_state=42)
print('Number of rows in train and validation data:', X_train.shape[0], y_train.shape)
print('Number of rows in validation data:', X_val.shape[0], y_val.shape)



def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 20
    lrate = initial_lrate * math.pow(drop,
    math.floor((1+epoch)/epochs_drop))
    return lrate


num_epochs = 1
batch_size = 1201

all_val_predictions = pd.DataFrame()

train_val_rmsle = []

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=X_train.shape[1]))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(264, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

lrate = LearningRateScheduler(step_decay)

model_hist = model.fit(X_train, y_train, validation_data=(X_val,y_val), 
                       batch_size=batch_size, epochs=num_epochs, verbose=1)

print('Getting model evaluation criteria...')

print('Train error is:', np.sqrt(mean_squared_error(y_train,model.predict(X_train))))

print('Validation error is:', np.sqrt(mean_squared_error(y_val,model.predict(X_val))))

print('Test error is:', np.sqrt(mean_squared_error(y_test,model.predict(scaler.transform(X_test)))))