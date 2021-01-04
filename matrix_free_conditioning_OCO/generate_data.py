#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 22:26:06 2020

@author: can
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random


def generate_2d_binary_class_data():
    np.random.seed(4)

    no_sample=450
    no_features=2
    noise=50
    #x1 
    x1=np.random.normal(2,1,[no_sample,no_features])
    x1=x1**2
    
    y1=np.ones(no_sample).reshape(no_sample,1)
    #x2 
    x2=np.random.normal(2,1,[no_sample,no_features])
    x2=-1*x2**2
    y2=(-1*np.ones(no_sample)).reshape(no_sample,1)
    
    
    #xnoise x1
    noise_x1=np.random.normal(2,1,[noise,no_features])
    noise_x1=noise_x1**2
    
    noise_y1=-1*(np.ones(noise)).reshape(noise,1)
    #noise x2
    noise_x2=np.random.normal(2,1,[noise,no_features])
    noise_x2=-1*noise_x2**2
    noise_y2=(np.ones(noise)).reshape(noise,1)
    
    
    
    
    
    tr_z=np.concatenate([x1,x2,noise_x1,noise_x2],axis=0)
    tr_y=np.concatenate([y1,y2,noise_y1,noise_y2],axis=0)
    
    
    random.seed(4)
    random_order=np.array([i for i in range(len(tr_z))])
    random.shuffle(random_order)
    
    tr_z=tr_z[random_order,]
    tr_y=tr_y[random_order,]
    
    return(tr_z,tr_y)