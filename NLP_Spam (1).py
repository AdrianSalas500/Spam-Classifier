#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().system('pip install tensorflow')
get_ipython().system('pip install tensorflow-text')


# In[8]:


import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[9]:


df=pd.read_excel('SMSSpamCollection.xlsx')
df = df.dropna(how = "any", axis = 1)


# In[10]:


df.columns = ['label', 'body']

df['label'].value_counts()


# We have 747 spam emails and 4826 ham emails. The ham messages are significantly higher, implying that 15% are spam emails and 85% of ham emails, indicating an imbalance, so in order to balance the two classes, we reduce number of ham messages to 747.

# First, I create two data frames, one for each class.

# In[11]:


df_spam = df[df['label']=='spam']
df_ham = df[df['label']=='ham']


# In[12]:


df_ham_balanced = df_ham.sample(df_spam.shape[0])


# In[13]:


df_balanced = pd.concat([df_ham_balanced, df_spam])


# In[14]:


df_balanced['label'].value_counts()


# Now the dataset is balanced.

# After balancing the data, we create another label representing if a message is spam (if it is 1) or ham (0). 

# In[15]:


df_balanced['spam'] = df_balanced['label'].apply(lambda x: 1 if x=='spam' else 0)


# In[16]:


df_balanced.sample(5)


# We will download two BERT models, one to perform preprocessing and the other one for encoding.

# In[17]:


bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")


# We are going to use preprocess as the input for this layer. Then, the encoder is going to convert the preprocessed text in vectors (output of the layer). 

# In[18]:


text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)


# Finally, this output is going to be fed in the neural network layers, that are two, the Dropout layer, and the Dense layer.

# In[19]:


layer = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
layer = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(layer)


# We add the input and output layers to construct the final model

# In[20]:


model = tf.keras.Model(inputs=[text_input], outputs = [layer])


# Then we include a model summary to see all the input and output layers that are used.

# In[21]:


model.summary()


# We are going to compile the model 

# In[22]:


loss = tf.keras.losses.BinaryCrossentropy()
metrics =  tf.keras.metrics.BinaryAccuracy(name='accuracy')

model.compile(optimizer='adam',
 loss=loss,
 metrics=metrics)


# Then we are going to fit the model. The model is going to learn from the samples of the training data, and identify patterns.

# In[23]:


X_train, X_test, y_train, y_test = train_test_split(df_balanced['body'],df_balanced['spam'], stratify=df_balanced['spam'])
model.fit(X_train, y_train, epochs=10)


# After training the model, we are going to predict and classify the samples in the testing dataset. We are going to get as an output an array of 0´s and 1´s, in which a 0 indicates that the message is ham and 1 if it is spam.

# In[24]:


y_pred = model.predict(X_test)
y_pred = y_pred.flatten()


# In[31]:


y_pred = np.where(y_pred > 0.5, 1, 0)
y_pred


# Also, we can make predictions inserting ourselves a set of messages, obtaining as a result an array of numbers, in which a number above 0.5 indicated that the message is considered spam, and a number below 0.5 that is ham.

# In[32]:


sample_dataset = [
 'You can win a lot of money, register in the link below',
 'You have an iPhone 10, spin the image below to claim your prize and it will be delivered in your door step',
 'You have an offer, the company will give you 50% off on every item purchased.',
 'Hey Bravin, do not be late for the meeting tomorrow will start lot exactly 10:30 am',
 "See you monday, we have alot to talk about the future of this company ."
]

model.predict(sample_dataset)

