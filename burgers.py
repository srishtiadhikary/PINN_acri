# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 11:31:59 2020

@author: Srishti Adhikary
"""

import tensorflow
import numpy as np
from tensorflow import keras
from tensorflow.keras import models,layers
 # dataset
x=np.linspace(-1,1,num=3000,endpoint=True,retstep=False,dtype='float64',axis=0)
t=np.linspace(0,1,num=3000,endpoint=True,retstep=False,dtype='float64',axis=0)


# neural network structure
pinn = tensorflow.keras.models.Sequential()
pinn.add(tensorflow.keras.layers.Dense(2,activation='tanh', kernel_initializer='variance_scaling'))
pinn.add(tensorflow.keras.layers.Dense(30,activation='tanh', kernel_initializer='variance_scaling'))
pinn.add(tensorflow.keras.layers.Dense(30,activation='tanh', kernel_initializer='variance_scaling'))
pinn.add(tensorflow.keras.layers.Dense(30,activation='tanh', kernel_initializer='variance_scaling'))
pinn.add(tensorflow.keras.layers.Dense(30,activation='tanh', kernel_initializer='variance_scaling'))
pinn.add(tensorflow.keras.layers.Dense(30,activation='tanh', kernel_initializer='variance_scaling'))
pinn.add(tensorflow.keras.layers.Dense(2,activation='tanh', kernel_initializer='variance_scaling')) 

# loss function
def lossfn(nn,xx,tt):
    def innloss(ucalc,upred):
        with tensorflow.GradientTape() as t1:
            inp=np.stack((xx,tt),axis=1)
            u=nn(inp)
            ux=t1.gradient(u,xx)
            ut=t1.gradient(u,tt)
            uxx=t1.gradient(ux,xx)
            f=ut+u*ux-(0.01/np.pi)*uxx
            ucalc = tensorflow.cast(ucalc, dtype='float64')
            upred = tensorflow.cast(upred, dtype='float64')
            loss1 = tensorflow.keras.losses.logcosh(upred,ucalc)
            loss2 = tensorflow.reduce_mean(tensorflow.square(f))
            return loss1 + loss2
    return innloss
        
#integrating the loss function
loss= lossfn(pinn,x,t)
pinn.compile(optimizer='Adam', loss=loss, metrics=[tensorflow.keras.metrics.LogCoshError(),tensorflow.keras.metrics.MeanSquaredError()])
pinn.fit(x, t, batch_size=30 ,epochs=1000)        
        
        
        
        
