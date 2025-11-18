#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Nov 30 09:11:12 2018

@author: philipp krah, jiahan wang
"""

###############################################################################
# IMPORTED MODULES
###############################################################################
import sys
sys.path.append('./../lib')
import numpy as np
from numpy import exp, mod,meshgrid, cos, sin, exp, pi
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sPOD_tools import shifted_rPCA, shifted_POD, give_interpolation_error
from transforms import transforms
from plot_utils import save_fig
###############################################################################

##########################################
#%% Define your DATA:
##########################################
plt.close("all")
method = "shifted_POD"
#method = "shifted_rPCA"
#case = "crossing_waves"
case = "sine_waves"
Nx = 100 # number of grid points in x
Nt = Nx//2  # numer of time intervalls
Niter = 1300 # number of sPOD iterations
Tmax = 0.5  # total time
L = 1  # total domain size
sigma = 0.015  # standard diviation of the puls
nmodes = 1  # reduction of singular values
x = np.arange(0,Nx)/Nx*L
t =np.arange(0,Nt)/Nt*Tmax
dx = x[1]-x[0]
dt = t[1]-t[0]
c = 1
[X, T] = meshgrid(x, t)
X = X.T
T = T.T

if case == "crossing_waves":
    fun = lambda x, t:  exp(-(mod((x-c*t), L)-0.1)**2/sigma**2) + \
                        exp(-(mod((x+c*t), L)-0.9)**2/sigma**2)

    # Define your field as a list of fields:
    # For example the first element in the list can be the density of
    # a flow quantity and the second element could be the velocity in 1D
    density = fun(X, T)
    velocity = fun(X, T)
    shifts1 = np.asarray(-c*t)
    shifts2 = np.asarray(c*t)
    fields = [density]  #, velocity]
elif case == "sine_waves":
    delta = 0.0125
    q1 = np.zeros_like(X)
    shifts1 = -0.25*cos(7*pi*t)
    for r in np.arange(1,5):
        x1 = 0.25 + 0.1 *r - shifts1
        q1 = q1 + sin(2*pi*r*T/Tmax)*exp(-(X-x1)**2/delta**2)
    c2 = dx/dt
    x2 = 0.2 + c2 * T
    shifts2 = -c2 * t
    q2 = exp(-(X-x2)**2/delta**2)
    fields = q1 + q2
    nmodes = [4,1]
#######################################
# %% CALL THE SPOD algorithm
######################################
data_shape = [Nx,1,1,Nt]
trafos = [transforms(data_shape ,[L], shifts = shifts1, dx = [dx], interp_order=5 ),
            transforms(data_shape ,[L], shifts = shifts2, dx = [dx], interp_order=5)]

interp_err = np.max([give_interpolation_error(fields,trafo) for trafo in trafos])
print("interpolation error: %1.2e "%interp_err)
# %%
qmat = np.reshape(fields,[Nx,Nt])
if method == "shifted_rPCA":
    mu = Nx * Nt / (4 * np.sum(np.abs(qmat)))*0.01
    lambd0 = 1 / np.sqrt(np.maximum(Nx, Nt)) * 1
    ret = shifted_rPCA(qmat, trafos, nmodes_max = np.max(nmodes)+10, eps=1e-16, Niter=Niter, use_rSVD=True, lambd=lambd0, mu = mu)
else:
    ret = shifted_POD(qmat, trafos, nmodes, eps=1e-16, Niter=Niter)
sPOD_frames, qtilde, rel_err = ret.frames, ret.data_approx, ret.rel_err_hist
qf = [np.squeeze(np.reshape(trafo.apply(frame.build_field()),data_shape)) for trafo,frame in zip(trafos,ret.frames)]
###########################################
# %% 1. visualize your results: sPOD frames
##########################################
# first we plot the resulting field
gridspec = {'width_ratios': [1, 1, 1, 1]}
fig, ax = plt.subplots(1, 4, figsize=(12, 4), gridspec_kw=gridspec,num=101)
mycmap = "viridis"
vmin = np.min(qtilde)*0.6
vmax = np.max(qtilde)*0.6

ax[0].pcolormesh(qmat,vmin=vmin,vmax=vmax,cmap=mycmap)
ax[0].set_title(r"$\mathbf{Q}$")
#ax[0].axis("image")
ax[0].axis("off")

ax[1].pcolormesh(qtilde,vmin=vmin,vmax=vmax,cmap=mycmap)
ax[1].set_title(r"$\tilde{\mathbf{Q}}$")
#ax[0].axis("image")
ax[1].axis("off")
# the result is a list of the decomposed field.
# each element of the list contains a frame of the decomposition.
# If you want to plot the k-th frame use:
# 1. frame
plot_shifted = True
k_frame = 0
if plot_shifted:
    ax[2].pcolormesh(qf[k_frame],vmin=vmin,vmax=vmax,cmap=mycmap)
    ax[2].set_title(r"$T^" + str(k_frame + 1) + "\mathbf{Q}^" + str(k_frame + 1) + "$")
else:
    ax[2].pcolormesh(sPOD_frames[k_frame].build_field())
    ax[2].set_title(r"$\mathbf{Q}^" + str(k_frame + 1) + "$")
ax[2].axis("off")
#ax[1].axis("image")
# 2. frame
k_frame = 1
if plot_shifted:
    im2 = ax[3].pcolormesh(qf[k_frame],vmin=vmin,vmax=vmax,cmap=mycmap)
    ax[3].set_title(r"$T^" + str(k_frame + 1) + "\mathbf{Q}^" + str(k_frame + 1) + "$")
else:
    im2 = ax[3].pcolormesh(sPOD_frames[k_frame].build_field())
    ax[3].set_title(r"$\mathbf{Q}^" + str(k_frame + 1) + "$")
ax[3].axis("off")
#ax[2].axis("image")

for axes in ax[:4]:
    axes.set_aspect(0.6)

plt.colorbar(im2)
plt.tight_layout()

save_fig("01_traveling_wave_1D_Frames.png",fig)
plt.show()
###########################################
# %% 2. visualize your results: relative error
##########################################

plt.figure(5)
plt.semilogy(rel_err)
plt.title("relative error")
plt.ylabel(r"$\frac{||X - \tilde{X}_i||_2}{||X||_2}$")
plt.xlabel(r"$i$")
plt.show()

