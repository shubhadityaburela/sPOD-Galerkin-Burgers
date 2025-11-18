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
sys.path.append('../lib/')
import numpy as np
from numpy import exp, mod, meshgrid, pi, sin, size, cos
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sPOD_tools import frame, shifted_POD, shifted_rPCA, build_all_frames
from transforms import transforms
from plot_utils import show_animation
from farge_colormaps import farge_colormap_multi
import random as random
###############################################################################
cm = farge_colormap_multi()
##########################################
#%% Define your DATA:
##########################################
plt.close("all")
Ngrid = [201, 202]  # number of grid points in x
Nt = 200            # Number of time intervalls
Nvar = 1            # Number of variables
nmodes = [2,3]          # reduction of singular values

data_shape = [*Ngrid,Nvar,Nt]
               # size of time intervall
T = 2*pi                # total time
dt = T/(Nt)
L = np.asarray([30, 30])   # total domain size
x0 = L*0.5               # starting point of the gauss puls
Nr_of_bumbs = [4,4]
delta = [2, 2]  # horizontal and vertical distance of bumbs
x,y = (np.linspace(-L[i]/2, L[i]/2, Ngrid[i]) for i in range(2))
time = np.arange(0, T, dt)
dx,dy = (x[1]-x[0],y[1]-y[0])
dt = time[1]-time[0]
c = dx/dt
[Y,X] = meshgrid(y,x)
saw = lambda x: x - np.floor(x)
#fun = lambda t: saw(t/2/pi)
fun = lambda t: sin(t)
#fun_n = lambda t: fun(t/2/pi*np.prod(Nr_of_bumbs))*np.heaviside()
b = lambda r: np.where(np.abs(r) < 1, np.exp(-1 / (1 - r ** 2)), 0)
#CoefList = [np.random.rand()) for n in range(20)];
#CoefList = [c/np.linalg.norm(c) for c in CoefList]
CoefList1 = np.exp(-0.5*np.arange(0,nmodes[0]))
CoefList2 = np.exp(-1*np.arange(0,nmodes[1]))

q = np.zeros(data_shape)
q1 = np.zeros(data_shape)
q2 = np.zeros(data_shape)
shift1, shift2, shift3 = np.zeros([2,Nt]),np.zeros([2,Nt]),np.zeros([2,Nt])

d = np.zeros([Ngrid[0],Nt])
center1 = (-0.2*L[0],0*L[1])
center2 = (0*L[0],0*L[1])

x0_list = [np.asarray([(ix) * delta[0],  (iy) * delta[0]]) for ix in range(-Nr_of_bumbs[0]//2,Nr_of_bumbs[0]//2) for iy in
               range(-Nr_of_bumbs[1]//2,Nr_of_bumbs[1]//2)]
At = np.sqrt(dt/(pi))
for it,t in enumerate(time):


    shift1[0,it] = 0
    shift1[1,it] = 0.3*L[1]*sin(t)

    shift2[0,it] = 0.3*L[1]*sin(t)
    shift2[1,it] = - 0.3*L[1]*sin(t)

    shift3[0,it] = 0
    shift3[1,it] = 0

    for k in range(np.prod(Nr_of_bumbs)):
        delta_x = center1[0] + x0_list[k][0] - shift1[0,it]
        delta_y = center1[1] + x0_list[k][1] - shift1[1,it]
        q1[..., 0, it] += At*fun(t / T * 2 * pi * k) * b(np.sqrt((X - delta_x)**2 +( Y - delta_y)**2)) * exp(-k/5)
    for k in range(4):
        delta_x = center2[0] + x0_list[k][0] - shift2[0, it]
        delta_y = center2[1] + x0_list[k][1] - shift2[1, it]
        q2[..., 0, it] += At*sin(t / T * 2 * pi * (k+1)) * b(np.sqrt((X - delta_x) ** 2 + (Y - delta_y) ** 2)) * exp(-k)

q = q1 + q2
plt.pcolormesh(X,Y,q[...,0,5])


# %% Create Trafo
import scipy.ndimage as ndimage

shift_trafo_1 = transforms(data_shape,L, shifts = shift1, dx = [dx,dy] , use_scipy_transform=True)
shift_trafo_2 = transforms(data_shape,L, shifts = shift2, dx = [dx,dy] , use_scipy_transform=True)
shift_trafo_3 = transforms(data_shape,L, trafo_type="identity", shifts = shift3, dx = [dx,dy] )


qshift1 = shift_trafo_1.reverse(q1)
qshift2 = shift_trafo_2.reverse(q2)
qshiftreverse = shift_trafo_2.apply(shift_trafo_2.reverse(q))
res = q-qshiftreverse
err = np.linalg.norm(np.reshape(res,-1))/np.linalg.norm(np.reshape(q,-1))
print("err =  %4.4e "% err)
plt.pcolormesh(X,Y,q[...,0,0]-qshiftreverse[...,0,0], cmap = cm, shading='auto')
plt.colorbar()
plt.show()
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
tranfos = [shift_trafo_1, shift_trafo_2]
#ret = shifted_POD(np.reshape(q,[-1,Nt]), transforms, nmodes=np.asarray(nmodes), eps=1e-4, Niter=50, visualize=True, use_rSVD=True)
qmat = np.reshape(q, [-1, Nt])
mu0 = np.size(qmat,0) * Nt / (4 * np.sum(np.abs(qmat)))
lambd0 =  1 / np.sqrt(np.max(Ngrid))
ret =  shifted_rPCA(qmat, tranfos, nmodes_max = np.prod(Nr_of_bumbs)+10, eps=1e-16, Niter=30, visualize=True, use_rSVD=True, mu = mu0/500, lambd = 2*lambd0)
qframes, qtilde , rel_err_list = ret.frames, ret.data_approx, ret.rel_err_hist
# %% Show results
U, S, VT = np.linalg.svd(np.reshape(q,[-1,Nt]),full_matrices=False)
U, S1, VT = np.linalg.svd(np.reshape(q1,[-1,Nt]),full_matrices=False)
U, S2, VT = np.linalg.svd(np.reshape(q2,[-1,Nt]),full_matrices=False)
S1f = qframes[0].modal_system["sigma"]
S2f = qframes[1].modal_system["sigma"]
plt.figure(3)
plt.semilogy(S/S[0], '--*', label = r"SVD $q$ ")
plt.semilogy(np.exp(-np.arange(0,16)/5),'-*', label = r"exact $q^1$ ")
plt.semilogy(np.exp(-np.arange(0,4)),'-o', label = r"exact $q^2$ ")
# plt.semilogy(S1/S1[0], '--*', label = r"exact $q^1$ ")
# plt.semilogy(S2/S2[0], '-o', label = r"exact $q^2$ ")
plt.semilogy(S1f/S1f[0], 'x', label = r"sPOD $q^1$")
plt.semilogy(S2f/S2f[0], '<', label = r"sPOD $q^2$")
plt.xlabel(r"rank $r$")
plt.ylabel(r"singular values $\sigma_r(q^k)/\sigma_0(q^k)$")
plt.xlim([-1,50])
plt.legend()


plt.figure(5)
nmode = 1
plt.plot(time, sin(time / T * 2 * pi * (nmode+1))*At,'-', label = r"exact $q^1$ " )
plt.plot(time, qframes[0].modal_system["VT"][nmode,:],'o', label = r"sPOD $q^1$")

U1f = qframes[0].modal_system["U"]
qshift1 = shift_trafo_1.reverse(q)
V1f = U1f.T@np.reshape(qshift1,[-1,Nt])
plt.plot(time, V1f[nmode,:]/np.sqrt(np.sum(V1f[nmode,:]**2)),'<',label=r"$\langle T^{-1}q,\phi_k^k(x)\rangle$" )
plt.legend()

# %%
plt.figure(4)
plt.semilogy(1-np.cumsum(S)/np.sum(S), '--*', label = r"SVD $q$ ")

qnorm = norm(qtilde,ord="fro")
rel_err_list = []
DoF_list = []
for r in range(1,np.max(nmodes)+10):
    qtilde = build_all_frames(qframes, tranfos, ranks=r)
    rel_err = norm(qmat-qtilde,ord="fro")/qnorm
    rel_err_list.append(rel_err)
    DoF_list.append( r*len(qframes))

plt.semilogy(DoF_list,rel_err_list, 'o', label = r"sPOD")
plt.legend()
plt.xlabel(r"$r$ DoFs")
plt.ylabel(r"rel err")


