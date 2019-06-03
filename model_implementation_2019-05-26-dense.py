
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


# In[7]:


clean_data.head()


# ### One-hot-encoding Brand Name

# In[8]:


unique_brand_names = pd.DataFrame(clean_data.clean_brand_name.value_counts())
min_brand_freq = 100
print('There are', unique_brand_names[unique_brand_names['clean_brand_name']>=min_brand_freq].shape[0],
      'brand names that occur >=',min_brand_freq, 'times in the dataset.','Will one-hot-encode these brands only.')


# In[9]:


clean_brand_name_df = clean_data['clean_brand_name']
clean_brand_name_df.shape


# In[10]:


clean_brand_name_df = pd.get_dummies(clean_brand_name_df)
clean_brand_name_df.shape


# In[11]:


drop_brand_col_lst = []
keep_brand_col_lst = list(unique_brand_names[unique_brand_names['clean_brand_name']>=min_brand_freq].index.values)
keep_brand_col_lst.remove('nobrandname')

for col_name in clean_brand_name_df.columns:
    if col_name not in keep_brand_col_lst:
        drop_brand_col_lst.append(col_name)

clean_brand_name_df_v2 =clean_brand_name_df.drop(columns=drop_brand_col_lst)
clean_brand_name_df_v2.columns = ['brand_' + str(col) for col in clean_brand_name_df_v2.columns]
clean_brand_name_df_v2.shape


# In[12]:


clean_data_v2 = pd.concat([clean_data.reset_index(drop=True)
                                   , clean_brand_name_df_v2.reset_index(drop=True)],
                                  axis=1).drop(columns=['brand_name','clean_brand_name'])
clean_data_v2.shape


# Delete the variables from memory

# In[13]:


del [[clean_brand_name_df,clean_brand_name_df_v2, clean_data]]
gc.collect()
clean_data = pd.DataFrame()
clean_brand_name_df=pd.DataFrame()
clean_brand_name_df_v2=pd.DataFrame()


# ### One-hot-encoding Assigned Category and Assigned Sub Category

# In[14]:


clean_category_name_df = clean_data_v2[['assigned_category', 'assigned_sub_category']]
clean_category_name_df.shape


# In[15]:


clean_category_name_df.head()


# In[16]:


clean_category_name_df_v2 = pd.get_dummies(clean_category_name_df)
clean_category_name_df_v2.shape


# In[17]:


# concat with main dataset on clean_category_name
print(clean_data_v2.shape)
clean_data_v3 = pd.concat([clean_data_v2, clean_category_name_df_v2], axis=1)
print(clean_data_v3.shape)


# In[18]:


# Delete dataframes from memory
del [[clean_category_name_df,clean_category_name_df_v2, clean_data_v2]]
gc.collect()
clean_data_v2 = pd.DataFrame()
clean_category_name_df=pd.DataFrame()
clean_category_name_df_v2=pd.DataFrame()


# ### One hot encode item condition

# In[19]:


item_condition_df = clean_data_v3[['item_condition_id']]
item_condition_df.shape


# In[20]:


item_condition_df['item_condition_id'] = item_condition_df['item_condition_id'].astype(object)


# In[21]:


item_condition_df_v2 = pd.get_dummies(item_condition_df)
item_condition_df_v2.shape


# In[22]:


get_ipython().run_cell_magic('time', '', 'print(clean_data_v3.shape)\nclean_data_v4 = pd.concat([clean_data_v3, item_condition_df_v2], axis=1)\nprint(clean_data_v4.shape)')


# In[23]:


# Delete dataframes from memory
del [[item_condition_df,item_condition_df_v2, clean_data_v3]]
gc.collect()
clean_data_v3 = pd.DataFrame()
item_condition_df_v2=pd.DataFrame()
item_condition_df=pd.DataFrame()


# ### Shipping
# - Change value of 0 to -1

# In[24]:


clean_data_v4['shipping'] = clean_data_v4['shipping'].replace([0], [-1])


# ### Drop Unwanted Columns

# In[28]:


clean_data_v4 = clean_data_v4.drop(columns=['name', 'category_name', 'item_description', 'stemmed_item_description',
       'clean_category_name', 'clean_item_name', 'assigned_category',
       'assigned_sub_category'])
clean_data_v4.shape


# # Model Implementation

# In[35]:


X_train, X_test, y_train, y_test = train_test_split(clean_data_v4.drop(columns=['price']).reset_index(drop=True), 
                                                    clean_data_v4[['price']].reset_index(drop=True), 
                                                                  test_size=0.1, random_state=42)
print('Number of rows in train and validation data:', X_train.shape[0], y_train.shape)
print('Number of rows in test data:', X_test.shape[0], y_test.shape)


# In[36]:


X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)


# In[37]:


scaler = MinMaxScaler(feature_range=(0, 1))
columns = X_train.columns
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=columns)
print('MinMaxScaler Complete')


# #### Model architecture parameters
# n_stocks = 500
# n_neurons_1 = 1024
# n_neurons_2 = 512
# n_neurons_3 = 256
# n_neurons_4 = 128
# n_target = 1

# In[38]:


X_train, X_val, y_train, y_val = train_test_split(X_train.reset_index(drop=True), 
                                                    y_train.reset_index(drop=True), 
                                                                  test_size=0.1, random_state=42)
print('Number of rows in train and validation data:', X_train.shape[0], y_train.shape)
print('Number of rows in validation data:', X_val.shape[0], y_val.shape)


# In[39]:


def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 20
    lrate = initial_lrate * math.pow(drop,
    math.floor((1+epoch)/epochs_drop))
    return lrate


# In[40]:


num_epochs = 1
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
model.compile(loss='mean_squared_error', optimizer='adam')

lrate = LearningRateScheduler(step_decay)

model_hist = model.fit(X_train, y_train, validation_data=(X_val,y_val), 
                       batch_size=batch_size, epochs=num_epochs, verbose=1)

# epoch_losses = pd.DataFrame(
#     [model_hist.history['loss'], model_hist.history['val_loss']], index=['train_loss', 
#                                                                     'validation_loss']).transpose()

# # plot the result of the epochs
# plt.plot(epoch_losses.index+1, epoch_losses.train_loss)
# plt.plot(epoch_losses.index+1, epoch_losses.validation_loss)
# plt.title('Epoch vs loss')
# plt.xlabel('Epcoh')
# plt.ylabel('Loss')
# plt.show()

# val_predictions = pd.concat([y_val.reset_index(drop=True),
#                          pd.DataFrame(model.predict(X_val), columns = ['price_val_predictions'])], axis=1)

# train_predictions = pd.concat([y_train.reset_index(drop=True),
#                          pd.DataFrame(model.predict(X_train), columns = ['price_tr_predictions'])], axis=1)

# all_val_predictions = pd.concat([all_val_predictions.reset_index(drop=True),
#                                 val_predictions.reset_index(drop=True)], axis=0)

# print('Train RMSLE:', round(rmsle(train_predictions['price'], train_predictions['price_tr_predictions']), 2),
#     'Validation RMSLE:', round(rmsle(val_predictions['price'], val_predictions['price_val_predictions']), 2))


# In[55]:


print('Train error is:', np.sqrt(mean_squared_error(y_train,model.predict(X_train))))

print('Validation error is:', np.sqrt(mean_squared_error(y_val,model.predict(X_val))))

print('Test error is:', np.sqrt(mean_squared_error(y_test,model.predict(scaler.transform(X_test)))))

