import tensorflow
import numpy as np
import random
from random import uniform
from tensorflow import keras
from tensorflow.keras import models,layers

 # dataset
xd=[]
td=[]
for i in range(3000):
    frand = random.uniform(-1, 1)
    xd.append(frand)
xdd=np.reshape(xd,(3000,1))
xdd=tensorflow.cast(xdd,dtype='float32')
for i in range(3000):
    frand = random.uniform(0, 1)
    td.append(frand)
tdd=np.reshape(td,(3000,1))
tdd=tensorflow.cast(tdd,dtype='float32')
xbc=random.choices([-1,1], k=3000)
xbcc=np.reshape(xbc,(3000,1))
xbcc=tensorflow.cast(xbcc,dtype='float32')
inpd=tensorflow.stack([xdd,tdd],axis=1)
inpbc=tensorflow.stack([xbcc,tdd],axis=1)
inp=tensorflow.stack([inpd,inpbc],axis=0)
x= inp[:,0:1]
t= inp[:,1:2]
out=np.zeros((3000,1),dtype='float32')
# neural network structure
x_in = tensorflow.keras.layers.Input(shape=(1), name="Position")
t_in = tensorflow.keras.layers.Input(shape=(1), name="Time")
IN = tensorflow.keras.layers.Concatenate(axis=-1, name="Input")([x_in, t_in])
dense1=(tensorflow.keras.layers.Dense(50,activation='tanh'))(IN)
dense2=(tensorflow.keras.layers.Dense(50,activation='tanh'))(dense1)
dense3=(tensorflow.keras.layers.Dense(50,activation='tanh'))(dense2)
dense4=(tensorflow.keras.layers.Dense(50,activation='tanh'))(dense3)
dense5=(tensorflow.keras.layers.Dense(50,activation='tanh'))(dense4)
output=(tensorflow.keras.layers.Dense(1,activation='tanh'))(dense5)
pinn = tensorflow.keras.Model(inputs=[x_in,t_in], outputs=output, name="pinn")


#loss function
def lossfn(nn, x, t):
  def loss(uac, ucalc):
    with tensorflow.GradientTape(persistent=True) as t1:
      t1.watch(x)
      t1.watch(t)
      with tensorflow.GradientTape() as t2:
        t2.watch(x)
        u = nn([x, t])
        ux = t2.gradient(u, x)
      ut = t1.gradient(u, t)
      uxx = t1.gradient(ux, x)
    f = ut + u*ux - (0.01/np.pi)*uxx
    ucalc = tensorflow.cast(ucalc, dtype='float32')
    uac = tensorflow.cast(uac, dtype='float32')      
    loss1 = tensorflow.reduce_mean(tensorflow.square(f))
    loss2 = tensorflow.reduce_mean(tensorflow.square(uac-ucalc))
    return loss1 + loss2
  return loss

    
        
#integrating the loss function
loss = lossfn(pinn, x, t)
pinn.compile(optimizer='adam', loss=loss, metrics=['mse'])
pinn.fit([inpbc[:,0:1], inpbc[:,1:2]], out, batch_size=30, epochs=30)
