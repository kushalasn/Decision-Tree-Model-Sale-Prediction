#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('train.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isna().sum()


# In[6]:


df.describe()


# In[7]:


df['LOYALTY_PROGRAM'].value_counts()


# In[8]:


df['OCCUPATION'].value_counts()


# In[9]:




# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(df[['PAST_PURCHASE']])

# Run the normalizer on the dataframe
df_normalized = pd.DataFrame(x_scaled)


# In[10]:


df_normalized.columns = ['PAST_PURCHASE_Normalised']


# In[11]:


df_sc1 = pd.concat([df, df_normalized], axis=1)


# In[12]:


df_sc1=df_sc1.drop(['PAST_PURCHASE'], axis=1)


# In[13]:


min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor
x_scaled1 = min_max_scaler.fit_transform(df[['AGE']])

# Run the normalizer on the dataframe
df_normalized1 = pd.DataFrame(x_scaled1)


# In[14]:


df_normalized1.columns = ['AGE_Normalised']
df_sc11 = pd.concat([df_sc1, df_normalized1], axis=1)
df_sc11=df_sc11.drop(['AGE'], axis=1)


# In[15]:


df_sc11.head()


# In[16]:


df_sc11.loc[:, "CUSTOMER_SINCE"] = df["CUSTOMER_SINCE"].apply(lambda x:2021-x)


# In[17]:


df_sc11


# In[18]:


df_sc11['PURCHASE'].isna().sum()


# In[19]:


#df_sc11.loc[:, "PURCHASE"] = df["PURCHASE"].apply({'No':0,'Yes':1})
df_sc11["PURCHASE"] = pd.get_dummies(df_sc11["PURCHASE"])


# In[20]:


df_sc11["PURCHASE"]


# In[21]:


df_sc11.tail()


# In[22]:


df_sc11['INCOME_GROUP'].value_counts()


# In[23]:


cleanup_nums = {"INCOME_GROUP": {"Low": 0, "High": 2, "Medium": 1, " ": 3}}
df_sc11 = df_sc11.replace(cleanup_nums)


# In[24]:


df_sc11['INCOME_GROUP'].value_counts()


# In[ ]:





# In[25]:


cleanup_nums = {"OCCUPATION": {"Salaried": 3, "Self employed": 1, "Business": 2, " ": 0}}
df_sc11 = df_sc11.replace(cleanup_nums)


# In[26]:


df_sc11


# In[27]:


df_sc11['LOYALTY_PROGRAM'].value_counts()


# In[28]:


df_sc11


# In[29]:


cleanup_nums = {"LOYALTY_PROGRAM": {"No": 0, "Yes": 1}}
df_sc11 = df_sc11.replace(cleanup_nums)


# In[30]:


df_sc11=df_sc11.dropna()


# In[31]:


sns.barplot(data=df_sc11, x='AGE_Normalised', y='PURCHASE', color='g')
plt.show()


# In[32]:


sns.jointplot(data=df_sc11, x='AGE_Normalised', y='PAST_PURCHASE_Normalised', kind='reg', color='g')
plt.show()


# In[33]:


df_sc11.loc[:, "AGE_Normalised"] = df_sc11["AGE_Normalised"].apply(lambda x:1-x)


# In[34]:


feature_cols = ['OCCUPATION', 'INCOME_GROUP', 'CUSTOMER_SINCE', 'LOYALTY_PROGRAM','PAST_PURCHASE_Normalised','AGE_Normalised']
x = df_sc11[feature_cols] # Features

y= df_sc11.PURCHASE


# In[35]:






from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.33, random_state=42)
from sklearn.tree import DecisionTreeClassifier
model= DecisionTreeClassifier()

model.fit(x_train,y_train)
y_pred=model.predict(x_test)









# In[36]:









from sklearn.metrics import classification_report,confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
sns.heatmap(cm,annot=True)
clf_report=classification_report(y_test,y_pred)
print(classification_report(y_test,y_pred))


# In[37]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[38]:


print('Testing Dataset  the model--------------------------------------')


# In[39]:


test=pd.read_csv('testing.csv')
test = pd.DataFrame(test)
test.head()
len(test)


# In[40]:



test.columns


# In[41]:


test=test.drop(['ID','STATE','Unnamed: 0'], axis = 1)
len(test)


# In[42]:




#Predict the response for test dataset
y_pred_test = model.predict(test)
y_pred_test


# In[43]:


len(y_pred_test)


# In[44]:


len(test)


# In[45]:


df_teaser=pd.DataFrame(y_pred_test)


# In[46]:


df_teaser.rename(columns = {0:'PURCHASE'}, inplace = True)


# In[47]:


df_teaser.value_counts()


# In[48]:


df_teaser


# In[49]:


data_not=pd.read_csv('sample_submission.csv')


# In[50]:


data_not


# In[51]:


data_not


# In[52]:


data_not=pd.concat([data_not,df_teaser],axis=1)


# In[53]:


data_not


# In[54]:


data_not.drop('PURCH',
  axis='columns', inplace=True)


# In[55]:



cleanup_nums = {"PURCHASE":     {0: 'No', 1: 'Yes'}}
data_not =data_not.replace(cleanup_nums)


# In[56]:


data_not.to_csv("Final_Submission_File.csv")


# In[57]:


data_not


# In[62]:


import pickle
pickle.dump(model,open('model.pkl','wb'))
print(test)


# In[67]:


model=pickle.load(open('model.pkl','rb'))
print(model.predict([test.loc[9900]]))


# In[68]:


print(test.loc[7])


# In[ ]:




