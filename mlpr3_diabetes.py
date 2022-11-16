#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

df = pd.read_csv("C:/Users/HP/Desktop/diabetes.csv")

df.head()

df.shape

df.columns

df.isna().sum()


X = df.drop( ["Outcome"], axis=1) 

y=df["Outcome"]

X.shape

y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

knn=KNeighborsClassifier (n_neighbors=3)

knn.fit(X_train, y_train)
                                                
                                                    
y_pred=knn.predict(X_test)
                                                    
# accuracy score

metrics.accuracy_score (y_test, y_pred)

from sklearn.metrics import confusion_matrix
                                                    
                                                    


#extracting true positives, false positives, true negatives, false negatives

print(confusion_matrix(y_test, y_pred)) 
tn, fp, fn, tp = confusion_matrix (y_test, y_pred).ravel() 


print("False Positives: ", fp) 
print("True Positives: ",tp)


print("True Negatives: ",tn)

print("False Negatives: ", fn)

#accuracy

Accuracy =(tn+tp)*100/(tp+tn+fp+fn) 

print("Accuracy {:0.2f}%:".format(Accuracy))
#Precision

Precision = tp/(tp+fp) 

print("Precision {:0.2f}".format(Precision))
                                                    
#Recall

Recall=tp/(tp+fn)

print("Recall {:0.2f}".format(Recall))
                                                    
#Error rate

err =(fp+fn)/(tp+tn + fn + fp)

print("Error rate (:0.2f)".format(err))


# In[2]:


import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

df = pd.read_csv("C:/Users/HP/Desktop/diabetes.csv")

df.head()


# In[4]:


df.shape



# In[5]:


df.columns


# In[6]:


df.isna().sum()


# In[7]:


X = df.drop( ["Outcome"], axis=1) 

y=df["Outcome"]


# In[8]:


X.shape


# In[9]:


y.shape


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)




# In[12]:


knn=KNeighborsClassifier (n_neighbors=3)


# In[15]:


knn.fit(X_train, y_train)


# In[18]:


y_pred=knn.predict(X_test)


# In[19]:


# accuracy score

metrics.accuracy_score (y_test, y_pred)


# In[21]:


from sklearn.metrics import confusion_matrix
                                                    
                                                    


#extracting true positives, false positives, true negatives, false negatives

print(confusion_matrix(y_test, y_pred)) 
tn, fp, fn, tp = confusion_matrix (y_test, y_pred).ravel() 


print("False Positives: ", fp) 
print("True Positives: ",tp)


print("True Negatives: ",tn)

print("False Negatives: ", fn)


# In[23]:


#accuracy

Accuracy =(tn+tp)*100/(tp+tn+fp+fn) 

print("Accuracy {:0.2f}%:".format(Accuracy))


# In[26]:


#Precision

Precision = tp/(tp+fp) 

print("Precision {:0.2f}".format(Precision))


# In[27]:


#Recall

Recall=tp/(tp+fn)

print("Recall {:0.2f}".format(Recall))


# In[29]:


#Error rate

err =(fp+fn)/(tp+tn + fn + fp)

print("Error rate {:0.2f}".format(err))


# In[ ]:




