# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 11:31:59 2020

@author: Srishti Adhikary
"""

import tensorflow
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
 # dataset
x=np.linspace(-1,1,num=3000,endpoint=True,retstep=False,dtype='float64',axis=0)
t=np.linspace(0,1,num=3000,endpoint=True,retstep=False,dtype='float64',axis=0)


# neural network structure
pinn = tensorflow.keras.model.Sequential()
pinn.add(tensorflow.keras.layers.Dense(2,activation='tanh', kernel_initializer='variance_scaling'))
pinn.add(tensorflow.keras.layers.Dense(30,activation='tanh', kernel_initializer='variance_scaling'))
pinn.add(tensorflow.keras.layers.Dense(30,activation='tanh', kernel_initializer='variance_scaling'))
pinn.add(tensorflow.keras.layers.Dense(30,activation='tanh', kernel_initializer='variance_scaling'))
pinn.add(tensorflow.keras.layers.Dense(30,activation='tanh', kernel_initializer='variance_scaling'))
pinn.add(tensorflow.keras.layers.Dense(30,activation='tanh', kernel_initializer='variance_scaling'))
pinn.add(tensorflow.keras.layers.Dense(2,activation='tanh', kernel_initializer='variance_scaling')) 

# loss function
def lossfn(nn,xx,tt):
    with tensorflow.GradientTape() as t1:
        inp=np.stack((xx,tt),axis=1)
        u=nn(inp)
        ux=t1.gradient(u,xx)
        ut=t1.gradient(u,tt)
        uxx=t1.gradient(ux,xx)
        f=ut+u*ux-(0.01/np.pi)*uxx
        
        
        