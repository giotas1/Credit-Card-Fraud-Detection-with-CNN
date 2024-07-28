#!/usr/bin/env python
# coding: utf-8

# # Step 1: Libraries

# In[44]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# # Step 2: Importing the dataset

# In[19]:


dataset_1  = pd.read_csv('creditcard.csv')


# In[20]:


dataset_1.head()


# # Step 3: Data Preprocessing

# In[21]:


dataset_1.shape


# In[22]:


# checking the null values
dataset_1.isnull().sum()


# In[23]:


dataset_1.info()


# In[24]:


# observations in each class
dataset_1['Class'].value_counts()


# In[25]:


# balence the dataset
fraud = dataset_1[dataset_1['Class']==1]
non_fraud = dataset_1[dataset_1['Class']==0]


# In[26]:


fraud.shape, non_fraud.shape


# In[27]:


# random selection of samples
non_fraud_t = non_fraud.sample(n=492)


# In[28]:


non_fraud_t.shape


# In[29]:


# merge dataset
dataset = pd.concat([fraud, non_fraud_t], ignore_index=True)


# In[30]:


print(dataset)


# In[31]:


# observations in each class
dataset['Class'].value_counts()


# In[32]:


# matrix of features
x = dataset.drop(labels=['Class'], axis=1)


# In[33]:


# dependent variable
y = dataset['Class']


# In[34]:


x.shape, y.shape


# In[35]:


# splitting the dataset into train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[36]:


x_train.shape, x_test.shape


# In[37]:


# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[38]:


x_train


# In[39]:


y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# In[40]:


x_train.shape, x_test.shape


# In[41]:


# reshape the dataset
x_train = x_train.reshape(787, 30, 1)
x_test = x_test.reshape(197, 30, 1)


# In[42]:


x_train.shape, x_test.shape


# # Step 4: Building the model

# In[45]:


# defining an object
model = tf.keras.models.Sequential()


# In[46]:


# first CNN layer
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=2, padding='same', activation='relu', input_shape = (30, 1)))

# batch normalization
model.add(tf.keras.layers.BatchNormalization())

# maxpool layer
model.add(tf.keras.layers.MaxPool1D(pool_size=2))

# dropout layer
model.add(tf.keras.layers.Dropout(0.2))


# In[47]:


# second CNN layer
model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, padding='same', activation='relu'))

# batch normalization
model.add(tf.keras.layers.BatchNormalization())

# maxpool layer
model.add(tf.keras.layers.MaxPool1D(pool_size=2))

# dropout layer
model.add(tf.keras.layers.Dropout(0.3))


# In[48]:


# flatten layer
model.add(tf.keras.layers.Flatten())


# In[49]:


# first dense layer
model.add(tf.keras.layers.Dense(units=64, activation='relu'))

# dropout layer
model.add(tf.keras.layers.Dropout(0.3))


# In[50]:


# output layer
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# In[51]:


model.summary()


# In[52]:


opt = tf.keras.optimizers.Adam(learning_rate=0.0001)


# In[53]:


model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


# # Step 5: Training the model

# In[54]:


history = model.fit(x_train, y_train, epochs=25, validation_data=(x_test, y_test))


# In[55]:


# model predictions
y_pred = model.predict(x_test)


# In[56]:


print(y_pred[12]), print(y_test[12])


# In[61]:


y_test = np.random.randint(0, 10, size=(100,))  # Example true labels
y_pred_probs = np.random.rand(100, 10)          # Example predicted probabilities
y_pred = np.argmax(y_pred_probs, axis=1)


# In[62]:


# confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[63]:


cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[64]:


acc_cm = accuracy_score(y_test, y_pred)
print(acc_cm)


# # Step 6: Learning Curve

# In[65]:


def learning_curve(history, epoch):

  # training vs validation accuracy
  epoch_range = range(1, epoch+1)
  plt.plot(epoch_range, history.history['accuracy'])
  plt.plot(epoch_range, history.history['val_accuracy'])
  plt.title('Model Accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'val'], loc='upper left')
  plt.show()

  # training vs validation loss
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Model Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'val'], loc='upper left')
  plt.show()


# In[66]:


learning_curve(history, 25)


# In[ ]:




