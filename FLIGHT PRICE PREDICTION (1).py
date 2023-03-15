#!/usr/bin/env python
# coding: utf-8

# In[164]:


import numpy as np
import pandas as pd
import pickle


# In[165]:


df = pd.read_csv("Clean_Dataset (1).csv")


# In[166]:


df.dropna()


# In[167]:


from numpy.random import default_rng

arr_indices_top_drop = default_rng().choice(df.index, size=295000, replace=False)
df2 = df.drop(index=arr_indices_top_drop)


# In[168]:


df2 = df2.dropna()


# In[169]:


df2.pop("Unnamed: 0")
df2.pop("flight")
df2.pop("duration")
df2.pop("days_left")
price = df2.pop("price")
price


# In[170]:


df2.head(5)


# In[171]:


from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
df1 = pd.DataFrame()
df1['airline'] = le.fit_transform(df2['airline'])
df1['source_city'] = le.fit_transform(df2['source_city'])
df1['departure_time'] = le.fit_transform(df2['departure_time'])
df1['stops'] = le.fit_transform(df2['stops'])
df1['arrival_time'] = le.fit_transform(df2['arrival_time'])
df1['destination_city'] = le.fit_transform(df2['destination_city'])
df1['class'] = le.fit_transform(df2['class'])



# In[187]:


from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(df1,price ,
                                   random_state=104, 
                                   test_size=0.25, 
                                   shuffle=True)





# In[191]:


from sklearn.ensemble import RandomForestClassifier  
classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")  
classifier.fit(X_train, y_train)  


# In[192]:


y_pred = classifier.predict(X_test)
pickle.dump(classifier, open('model2.pkl','wb'))


