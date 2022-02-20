#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns


# In[3]:


data = pd.read_csv(r'C:\Users\nivet\Downloads\archive (3)\insurance.csv')


# In[4]:


data.head()


# In[5]:


data.tail()


# ### Getting the shape of the dataset

# In[6]:


print(f'The number of rows and coulmns are {data.shape[0],data.shape[1]} respectively')


# ### Getting the gist of the dataset

# In[7]:


data.info()


# ### Check for null-values

# In[8]:


print(f'Are there any null values? {data.isnull().values.any()}')


# In[9]:


data.isnull().sum()


# In[10]:


sns.heatmap(data.isnull())


# ### Get the overall stats of the dataset

# In[11]:


data.describe()


# In[12]:


# to include stats for non-numerical columns as well
data.describe(include = 'all')


# ### Covert Columns From String ['sex','smoker', 'region' ] To Numerical Values

# In[13]:


# we need to convert because ML algorithms only process numerical values
data.head()


# In[14]:


data.sex.unique()


# In[15]:


data['sex'] = data['sex'].map({'female':0,'male':1})


# In[16]:


data['sex'].unique()


# In[17]:


data['smoker'].unique()


# In[18]:


data['smoker'] = data['smoker'].map({'no':0,'yes':1})


# In[19]:


data.head()


# In[20]:


data['region'].unique()


# In[21]:


data['region'] = data['region'].map({'southwest':1,'southeast':2,'northwest':3,'northeast':4})


# In[22]:


data.head()


# ### Store Feature Matrix In X and Response(Target) In Vector y

# In[23]:


data.columns


# In[24]:


# independant variables
X = data.drop(['charges'],axis = 1)


# In[25]:


X


# In[26]:


# Dependant variable
y = data['charges']


# In[27]:


y


# ### Train/Test split
#           1. Split data into two-part: a training set and a testing set
#           2. Train the model(s) on the training set
#           3. Test the Model(s) on the Testing set

# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### Import the models

# In[30]:


# This is a regression model as the values of the dependant variables are continuous
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# ### Model Training

# In[31]:


lr = LinearRegression()
lr.fit(X_train,y_train)
svm = SVR()
svm.fit(X_train,y_train)
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
gr = GradientBoostingRegressor()
gr.fit(X_train,y_train)


# ### Prediction on Test Data

# In[32]:


y_pred1 = lr.predict(X_test)
y_pred2 = svm.predict(X_test)
y_pred3 = rf.predict(X_test)
y_pred4 = gr.predict(X_test)

df1 = pd.DataFrame({'Actual':y_test,'Lr':y_pred1,'svm':y_pred2,'rf':y_pred3,'gr':y_pred4})


# In[33]:


df1


# ### Compare Performance visually

# In[34]:


import matplotlib.pyplot as plt


# In[35]:


# Visualizing 4 plots, Actual values vs the other values
plt.subplot(221)
plt.plot(df1['Actual'],label='Actual')
plt.plot(df1['Lr'],label='Lr')
plt.legend()


# In[36]:


# this is difficult to comprehend, so, comparing the first few rows
plt.subplot(221)
plt.plot(df1['Actual'].iloc[0:11],label='Actual')
plt.plot(df1['Lr'].iloc[0:11],label='Lr')
plt.legend()

plt.subplot(222)
plt.plot(df1['Actual'].iloc[0:11],label='Actual')
plt.plot(df1['svm'].iloc[0:11],label='svm')
plt.legend()


plt.subplot(223)
plt.plot(df1['Actual'].iloc[0:11],label='Actual')
plt.plot(df1['rf'].iloc[0:11],label='rf')
plt.legend()

plt.subplot(224)
plt.plot(df1['Actual'].iloc[0:11],label='Actual')
plt.plot(df1['gr'].iloc[0:11],label='gr')
plt.legend()
plt.tight_layout()


# In[37]:


# Findings: Model 1 and 4 are closer to the actual values than the others


# ### Evaluating the Algorithm

# In[38]:


from sklearn import metrics


# In[39]:


# Evaluating using r-square, it measures the goodness of fit.
# Greater the value, the better the model
score1 = metrics.r2_score(y_test,y_pred1)
score2 = metrics.r2_score(y_test,y_pred2)
score3 = metrics.r2_score(y_test,y_pred3)
score4 = metrics.r2_score(y_test,y_pred4)


# In[40]:


print(score1,score2,score3,score4,end="\n")


# In[41]:


# Findings : Model 4 is performing the best out of all


# In[42]:


# Using mean absolute error, the lower the value the better
s1 = metrics.mean_absolute_error(y_test,y_pred1)
s2 = metrics.mean_absolute_error(y_test,y_pred2)
s3 = metrics.mean_absolute_error(y_test,y_pred3)
s4 = metrics.mean_absolute_error(y_test,y_pred4)


# In[43]:


print(s1,s2,s3,s4,end='\n')


# In[44]:


# Findings : Model 4 is performing the best out of all


# ### Predict Charges for new customer

# In[45]:


data = {'age':40,
       'sex':1,
        'bmi':40.30,
        'children':4,
        'smoker':1,
        'region':2
       }
df = pd.DataFrame(data,index=[0])
df


# In[46]:


new_pred = gr.predict(df)
print(new_pred)


# ### Save Model Using Joblib

# In[47]:


# The model should be saved so that it doesn't have to be re-trained everytime its used
# Before deployment, the model has to be tested on entire dataset, not just x_train and y_train
# training of x and y_train are only done to evalute the model and figure out the best one


# In[48]:


# Training Gradient Boosting Model on entire dataset before deployment
gr = GradientBoostingRegressor()
gr.fit(X,y)


# In[49]:


import joblib


# In[50]:


joblib.dump(gr,'model_insurance')


# In[51]:


model = joblib.load('model_insurance')


# In[52]:


model.predict(df)


# In[53]:


# There are 2 different predictions as the models have been trained on the full DF as well as a portion of it


# ### GUI

# In[54]:


from tkinter import *


# In[55]:


import joblib


# In[56]:


window = Tk()
window.title('Insurance Cost Prediction')
label = Label(window, text = 'Insurance Cost Prediction', bg = 'black', fg = 'white').grid(row = 0,columnspan = 2)

def show_entry():
    p1 = float(e1.get())
    p2 = float(e2.get())
    p3 = float(e3.get())
    p4 = float(e4.get())
    p5 = float(e5.get())
    p6 = float(e6.get())
    
    model = joblib.load('model_insurance')
    result = model.predict([[p1,p2,p3,p4,p5,p6]])
    Label(window,text='Insurance Cost').grid(row = 7)
    Label(window,text = result).grid(row = 8)
    
Label(window,text = 'Enter Your Age').grid(row = 1)
Label(window,text = 'Male/Female [1/0]').grid(row = 2)
Label(window,text = 'Enetr BMI value').grid(row = 3)
Label(window,text = 'Enter number of children').grid(row = 4)
Label(window,text = 'Smoker [Yes/No][1/0]').grid(row = 5)
Label(window,text = 'Region [1-4]').grid(row = 6)

e1 = Entry(window)
e2 = Entry(window)
e3 = Entry(window)
e4 = Entry(window)
e5 = Entry(window)
e6 = Entry(window)

e1.grid(row = 1,column = 1)
e2.grid(row = 2,column = 1)
e3.grid(row = 3,column = 1)
e4.grid(row = 4,column = 1)
e5.grid(row = 5,column = 1)
e6.grid(row = 6,column = 1)

Button(window,text='Predict',command = show_entry).grid()

window.mainloop()


# In[ ]:





# In[ ]:




