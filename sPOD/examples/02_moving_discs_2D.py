#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D MOVING DISCS

Created on Sat Jan  2 14:52:26 2021

@author: phil
"""

###############################################################################
# IMPORTED MODULES
###############################################################################
import sys
sys.path.append('../lib')
import numpy as np
from numpy import exp, mod,meshgrid,pi,sin,size
import matplotlib.pyplot as plt
from sPOD_tools import frame, shifted_POD,shifted_rPCA
from transforms import transforms
from farge_colormaps import farge_colormap_multi
###############################################################################
cm = farge_colormap_multi()
##########################################
#%% Define your DATA:
##########################################
plt.close("all")
Ngrid = [401, 202]  # number of grid points in x
Nt = 50            # Number of time intervalls
Nvar = 1            # Number of variables
nmodes = 1          # reduction of singular values

data_shape = [*Ngrid,Nvar,Nt]
               # size of time intervall
T = 2*pi                # total time
L = np.asarray([2, 1])   # total domain size
x0 = L*0.5               # starting point of the gauss puls
R = 0.1 * min(L)         # Radius of cylinder
x,y = (np.linspace(0, L[i], Ngrid[i]) for i in range(2))
time = np.linspace(0, T, Nt)
dx,dy = (x[1]-x[0],y[1]-y[0])
dt = time[1]-time[0]
c = dx/dt
[Y,X] = meshgrid(y,x)


f = lambda x,l : ((np.tanh(x/l) + 1 ) * 0.5)

phi1 = np.zeros(data_shape[:-2])
phi2 = np.zeros(data_shape[:-2])
q = np.zeros(data_shape)

shift1 = np.zeros([2,Nt])
shift2 = np.zeros([2,Nt])

center1 = (0.2*L[0],0.5*L[1])
center2 = (0.5*L[0],0.5*L[1])

for it,t in enumerate(time):


    
    shift1[0,it] = 0
    shift1[1,it] = 0.3*L[1]*sin(t)
    
    shift2[0,it] = -0.3*L[0]*sin(t)
    shift2[1,it] = - 0.3*L[1]*sin(t)

    x1,y1 = (center1[0] - shift1[0,it],  center1[1] - shift1[1,it])
    x2,y2 = (center2[0] - shift2[0,it],  center2[1] - shift2[1,it])

    phi1 = np.sqrt((X-x1)**2 + (Y-y1)**2) - R
    phi2 = np.sqrt((X-x2)**2 + (Y-y2)**2) - R
    
    q[...,0,it] = f(phi1,dx*3)*f(phi2,dx*3) 
    #q[...,1,it] = f(phi1,dx)-f(phi2,dx) 
    #plt.pcolormesh(X,Y,q[...,1,it])
    #plt.show()
    #plt.pause(0.001)
    
    

# %% Create Trafo

shift_trafo_1 = transforms(data_shape,L, shifts = shift1, dx = [dx,dy], use_scipy_transform=False )
shift_trafo_2 = transforms(data_shape,L, shifts = shift2, dx = [dx,dy], use_scipy_transform=False )

qshift1 = shift_trafo_1.reverse(q)
qshift2 = shift_trafo_2.reverse(q)
qshiftreverse = shift_trafo_2.apply(shift_trafo_2.reverse(q))
res = q-qshiftreverse
err = np.linalg.norm(np.reshape(res,-1))/np.linalg.norm(np.reshape(q,-1))
print("err =  %4.4e "% err)
plt.pcolormesh(X,Y,q[...,0,0]-qshiftreverse[...,0,0],cmap = cm)
plt.colorbar()
# %% Test Trafo

# figs,axs = plt.subplots(3,1,sharex=True)
# axs[0].pcolormesh(X,Y,qshift1[...,0,0])
# axs[1].pcolormesh(X,Y,qshift2[...,0,0])
# axs[2].pcolormesh(X,Y,q[...,0,0]-qshiftreverse[...,0,0])
# axs[0].set_title(r"$q^1$ Frame 1")
# axs[1].set_title(r"$q^2$ Frame 2")
# axs[2].set_title(r"$q - T^s_1 q^1 + T^s_2 q^2$ Data")
# for it in range(Nt):
#     axs[0].pcolormesh(X,Y,qshift1[...,0,it])
#     axs[1].pcolormesh(X,Y,qshift2[...,0,it])
#     axs[2].pcolormesh(X,Y,q[...,0,it]-qshiftreverse[...,0,it])
#     plt.show()
#     plt.pause(0.001)
    
# %% Run shifted POD
transforms = [shift_trafo_1, shift_trafo_2]
qmat = np.reshape(q,[-1,Nt])
#qframes, q = shifted_POD(q, transforms, nmodes=2, eps=1e-4, Niter=20, visualize=True)
mu = np.prod(np.size(qmat)) / (4 * np.sum(np.abs(qmat)))*0.1
ret = shifted_rPCA(qmat, transforms, eps=1e-4, Niter=50, visualize=True, mu = mu)
qframes, qtilde = ret.frames, ret.data_approx
qtilde = np.reshape(qtilde,data_shape)
plt.pcolormesh(X,Y,q[...,0,10]-qtilde[...,0,10])
