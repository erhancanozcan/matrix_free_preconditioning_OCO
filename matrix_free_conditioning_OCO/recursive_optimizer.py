#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:13:15 2020

@author: can
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random
import cvxpy as cp
import os
from numpy import linalg as LA
os.chdir("/Users/can/Desktop/phd/Rotation1_Ashok_Cutkosky/code")




class recursive_optimizer_algo:
    
    
    def __init__(self,epsilon,tr_z,tr_y,eta,loss_type,intercept=True):
        self.epsilon=epsilon
        
        if intercept==True:
            ones=np.repeat(-1.0,len(tr_z)).reshape(len(tr_z),1)
            tr_z=np.concatenate((tr_z,ones),1)
        self.tr_z=tr_z
        self.tr_y=tr_y
        self.eta=eta
        #number_of_features in data set
        self.no_f=len(tr_z[0])
        
        self.wealth_algo1=np.array([epsilon])
        
        
        self.wealth_algo2=np.repeat(epsilon,self.no_f)
        self.A=np.repeat(5,self.no_f)
        self.v_i=np.zeros(self.no_f)
        self.sum_z_t=np.zeros(self.no_f)
        self.wealth_list=[]
        self.t=0
        self.loss_type=loss_type
        
    def calculate_gradient(self,tr_y,tr_z,x):
        if self.loss_type == "hinge":
            cntrl=(1 - tr_y * np.dot(tr_z,x.flatten()))[0]
            l_t_x=max(cntrl,0)
            #l_t_x_list.append(l_t_x)
            
            #print(cntrl)
            
            if cntrl < 0:
                g_t = np.zeros(self.no_f).reshape(self.no_f,1)
            elif cntrl == 0:
                alpha=0
                g_t= (-1 * alpha * tr_y * tr_z).reshape(self.no_f,1)
            else:
                g_t= (-1 * tr_y * tr_z).reshape(self.no_f,1)
                
            
            if(LA.norm(g_t,2))>0:
                g_t=g_t/LA.norm(g_t,2)
            
            self.gradient=g_t
            return(g_t)
            
        
    
    def optimizer1(self):
        self.x_i=self.wealth_algo2 * self.v_i
        self.w_t_i=np.clip(self.x_i,-0.5,0.5)
        
    def optimizer2(self):
        
        
        
        v_t_algo1=self.w_t_i.reshape(1,self.no_f)
        w_t_algo1=np.dot(self.wealth_algo1,v_t_algo1)
        
        #note that  w_t_algo1 represents weights for model. By using weights
        #at the corresponding iteration, I will calculate total loss in the 
        # 'run_algorithm' function.
        self.weights=w_t_algo1
        
        
        #we need to calculate g_t for the current data point.
        #Note that w_t_algo1 represents model weights.
        #t th data point is utilized to calculate gradient.
        #note that while calculating g_t, I am scaling it so that its length will
        #be at most 1.
        g_t_algo1=self.calculate_gradient(self.tr_y[self.t,:],self.tr_z[self.t,:],w_t_algo1)
        
        update=np.dot(g_t_algo1.reshape(1,self.no_f),w_t_algo1.T)
        
        self.wealth_algo1=self.wealth_algo1 - update
        
        self.z= g_t_algo1 / (1 - np.dot(g_t_algo1.reshape(1,self.no_f),v_t_algo1.T))
        
        
    def optimizer3(self):
    
        #after line 10 of algo2
        
        #g_t_algo2=z
        #g_t_algo2_tmp=g_t_algo2
        
        #g_t_algo2_tmp
        
        #if statement in line 13
        tmp_if_cntrl= self.z.flatten() * (self.x_i - self.w_t_i)
        self.z=self.z.flatten()
        self.z[tmp_if_cntrl<0]=0
        
        
        #update part
        self.wealth_algo2=self.wealth_algo2 - (self.x_i * self.z)
        
        z_t=self.z / (1 - (self.z * self.v_i))
        
        
        self.A = self.A + z_t**2
        
        self.sum_z_t=self.sum_z_t + z_t
        
        
        self.v_i=np.clip ( (-2*self.eta*self.sum_z_t) / self.A , -0.5, 0.5)
        self.t=self.t+1
    
        
        
    def run_algorithm(self):
    
        for t in range(len(self.tr_y)):
            self.optimizer1()
            self.optimizer2()
            self.wealth_list.append(self.wealth_algo1)
            self.optimizer3()
            
            #w_t_i,x_i=optimizer1(wealth_algo2,v_i)
            #z,wealth_algo1=optimizer2(w_t_i,wealth_algo1,g_t_algo1[t,:],no_f)
            
            
                
            
        