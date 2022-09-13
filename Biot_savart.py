# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 22:46:33 2021

@author: elraw
"""

import numpy as np 



def induced_vel(X1, X2, Xp, gamma = 1.0, core = 0.0, tol = 1e-6):
    
    x1 = X1[0,:];  y1 = X1[1,:] ; z1 = X1[2,:]
    x2 = X2[0,:];  y2 = X2[1,:] ; z2 = X2[2,:]
    xp = Xp[0,:];  yp = Xp[1,:] ; zp = Xp[2,:]
    
    R1 = np.sqrt((xp-x1)**2 + (yp-y1)**2 + (zp-z1)**2)
    R2 = np.sqrt((xp-x2)**2 + (yp-y2)**2 + (zp-z2)**2)
    
    R1_2x = (yp-y1)*(zp-z2) - (zp-z1)*(yp-y2)
    R1_2y = -(xp-x1)*(zp-z2) + (zp-z1)*(xp-x2)
    R1_2z = (xp-x1)*(yp-y2) - (yp-y1)*(xp-x2)
    R1_2sqrt = R1_2x**2 + R1_2y**2+ R1_2z**2
    R0_1 = (x2-x1)*(xp-x1)+(y2-y1)*(yp-y1)+(z2-z1)*(zp-z1) 
    R0_2 = (x2-x1)*(xp-x2) + (y2-y1)*(yp-y2)+(z2-z1)*(zp-z2)
    
    dir_v = (X2 - X1)/np.linalg.norm((X2 - X1), axis = 0);
    dir_cp = (Xp - X1)/np.linalg.norm((Xp - X1), axis = 0);

    boolien = np.linalg.norm(np.cross(dir_v, dir_cp, axisa = 0, axisb = 0).T, axis = 0);

    # boolien = np.sum(np.absolute(dir_v - dir_cp) > tol*np.ones_like(dir_v), axis = 0);

    # R1_2sqrt[R1_2sqrt<core**2] = core**2
    # R1[R1<core] = core
    # R2[R2<core] = core 
    K = np.zeros_like(boolien);
    idx = boolien > tol;
    K[idx] = gamma/(4*np.pi*R1_2sqrt[idx])*(R0_1[idx]/R1[idx] - R0_2[idx]/R2[idx]);
    U = K*R1_2x; V = K*R1_2y; W = K*R1_2z;
    
    return U,V,W  

if __name__ == "__main__":
    X1 = np.array([[0, 0], [0, 0], [0, 1]])
    X2 = np.array([[0, 0], [0, 0], [1, 0]])
    Xp = np.array([[0],[0],[0.5]])
    gamma = 1 
    u,v,w = induced_vel(X1, X2, Xp, gamma)
    


# X1 = np.array([[0], [0], [0]])
# X2 = np.array([[-1], [ 1], [5]])
# Xp = np.array([[0.5],[0.5],[0]])
# gamma = 1 
# u,v,w = induced_vel(X1, X2, Xp, gamma)
