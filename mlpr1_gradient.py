#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

def gradientDescent(start,gradient,learningrate,iteration,tol=0.01):
    
    steps=[start]
    X = start
    for i in range(iteration):
        difference = learningrate*gradient(X)
        if np.abs(difference)<tol:
            break
        X=X-difference
        steps.append(X)
        
    return steps,learningrate,X,len(steps)

def gradient_fun(X):
    return (X)**2*6*X+9

def gradient(X):
    return(2*X+6)

history,learningrate,result,steps= gradientDescent(2,gradient_fun,0.001,100)

print("steps in GD",history)
print("learning rate",learningrate)
print("no of steps for local minima",steps)


# In[ ]:




