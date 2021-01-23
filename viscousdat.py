# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 20:42:21 2021

@author: Srishti Adhikary
"""
import numpy as np
import random
import csv
xd=[]
td=[]
for i in range(5000):
    frand = random.uniform(-1, 1)
    xd.append(frand)
xdd=np.reshape(xd,(5000,1))

for i in range(5000):
    frand = random.uniform(0, 1)
    td.append(frand)
tdd=np.reshape(td,(5000,1))
nu=0.02
def u(x,t):
    u=0.5-0.5*np.tanh((x-0.5*t)/(4*nu))
    return u
uu=u(xdd,tdd)
myData = [xdd,tdd,uu]
myFile = open('vis2.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(myData)
