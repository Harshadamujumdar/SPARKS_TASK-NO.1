

# THE SPARKS FOUNDATION 
# 
# DATA SCIENCE &  BUSINESS ANALYTICS TASKS
# 
# TASK1: Prediction Using Supervised ML
# 
# Harshada Rajvardhan Mujumdar

# OBJECTIVE: Prdict the percentage of student based on number of hours studied

# Imporatant Required libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Dataset

# In[3]:


url = "http://bit.ly/w-data"
data = pd.read_csv(url)
data.head(10)


# In[4]:


# checking the number of rows and columns in the dataset
data.shape


# In[5]:


# checking for the number of elements in the dataset
data.size


# In[6]:


# type information of the file
data.info()


# Checking for the null values in data

# In[7]:


# to identify the no of missing data
data.isnull().sum()


# Checking for mean, median, maximum, minimum values of the dataset

# In[8]:


data.describe()


# Plotting the data on graph

# In[9]:


data.plot(x = 'Hours', y = 'Scores', style = 'o')
plt.title('Hours vs Percentage')
plt.xlabel('No. of hours studied')
plt.ylabel('Percentage Score')

plt.show()


# The above graph shows that as the number of hours increases the percentage of the each student is also increased. The plot is positively linear.

# Preparing the data

# In[11]:


# accessing rows and columns by index-based
X = data.iloc[:, :-1].values
print(X)


# In[12]:


y= data.iloc[:,1].values
print(y)


# In[13]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)


# Training the Algorithm

# In[14]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[15]:


regressor.coef_,regressor.intercept_


# In[16]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# Making Prediction

# In[17]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[18]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# In[19]:


hours = np.array([9.25])
hours_studied = hours.reshape(-1,1)
own_pred = regressor.predict(hours_studied)


# In[20]:


print('If a student studied 9.25 hrs/day the score would be:', own_pred[0])


# Evaluating the model

# In[21]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))

