# -*- coding: utf-8 -*-
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
sys.path.append('../lib')
from numpy import *
from numpy.linalg import svd,norm
import numpy as np
from numpy import exp, mod,meshgrid
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
###############################################################################
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def remap(phi, x0, y0, alpha):
    pass
    

N = np.asarray([100, 101])
L = np.asarray([1, 1])
delta = 0.02    # peak at the midpoint of the soft field
lambd = 0.01    # thickness of the step
R = 0.25        # radius of the cylinder
Nmodes=10
x=[]
alpha0 = np.asarray([0.5 , 1])
x.append(np.linspace(0,L[0],N[0]))
x.append(np.linspace(0,L[1],N[1]))

[X,Y] = np.meshgrid(x[1],x[0])

n_snapshots = 25;

phis = np.zeros([N[0] ,N[1], n_snapshots])
qs = np.zeros([N[0] ,N[1], n_snapshots])

f = lambda x,l : ((np.tanh(x/l) + 1 ) * 0.5)

fig = plt.figure(0)
fig.clf()
ax=fig.gca()
for k in range(n_snapshots):
    z = np.exp(1j*2*np.pi *k/n_snapshots)
    x0= 0.5 + R*np.real(z)
    y0= 0.5 + R*np.imag(z)
    alpha = [alpha0[0]*(np.real(z)+0.1), alpha0[1]*(np.imag(z)+0.2)]
    phi = sqrt(((X-x0)/alpha[0])**2 + ((Y-y0)/alpha[1])**2 + delta**2) -0.15 -delta;
    
    q   = f(phi,lambd); 
    
    phis[:,:,k] = phi ;    # store distance function 
    qs[:,:,k] = q   ;    # store field 
    
    if 1:
        plt.subplot(121)
        plt.pcolor(X,Y,phi)
        plt.title("$\phi$")
        plt.ylabel("$x$")
        plt.xlabel("$y$")
        plt.axis([0, L[0], 0, L[1]])
        plt.gca().set_aspect('equal', adjustable='box')
        
        plt.subplot(122)
        plt.title("$f(\phi)$")
        plt.pcolor(X,Y,q)
        plt.ylabel("$x$")
        plt.xlabel("$y$")
        plt.axis([0 ,L[0] ,0, L[1]])
        plt.gca().set_aspect('equal', adjustable='box')

        #plt.axis("equal")
        #plt.xlim([0,L[0]])
        #plt.ylim([0,L[1]])
        plt.pause(0.01)
        plt.show()
        plt.savefig("quenched_ball"+str(k).zfill(3)+".png")
        
# %%
[Uphi,Sphi,Vphi] = svd(reshape(phis,[-1,n_snapshots]), full_matrices=True)
Sphi = Sphi[:Nmodes]
Uphi = Uphi[:, :Nmodes]
Vphi = Vphi[:Nmodes, :]

[Uq,Sq,Vq] = svd(reshape(qs,[-1,n_snapshots]),full_matrices=True)
Sq = Sq[:Nmodes]
Uq = Uq[:, :Nmodes]
Vq = Vq[:Nmodes, :]

        
# %%
ntime=2
phi_appr=np.reshape(np.dot(Uphi * Sphi, Vphi), [N[0],N[1],n_snapshots])
fphi=f(phi_appr,lambd)
q_appr=np.reshape(np.dot(Uq * Sq, Vq), [N[0],N[1],n_snapshots])



for ntime in range(n_snapshots):
    fig = plt.figure(0)
    fig.clf()
    h=[]

    plt.suptitle("Snapshot:" + str(ntime))
    plt.subplot(221)
    plt.pcolor(X,Y,fphi[:,:,ntime])
    plt.title("$f(\\tilde{\phi})$")
    plt.ylabel("$x$")
    plt.xlabel("$y$")
    plt.axis("equal")
    plt.subplot(223)
    h1 = plt.pcolor(X,Y,fphi[:,:,ntime]-qs[:,:,ntime])
    plt.title("$f(\\tilde{\phi})-q$")
    plt.ylabel("$x$")
    plt.xlabel("$y$")
    plt.colorbar()
    plt.axis("equal")
    
    plt.subplot(222)
    plt.title("$\\tilde{q}$")
    plt.pcolor(X,Y,q_appr[:,:,ntime])
    plt.ylabel("$x$")
    plt.xlabel("$y$")
    plt.axis("equal")
    
    plt.subplot(224)
    h2 = plt.pcolor(X,Y,q_appr[:,:,ntime]-qs[:,:,ntime])
    plt.title("$\\tilde{q}-q$")
    plt.ylabel("$x$")
    plt.xlabel("$y$")
    plt.colorbar()
    cmin = min([h1.get_clim()[0], h2.get_clim()[0]])
    cmax = max(h1.get_clim()[1], h2.get_clim()[1])
    h1.set_clim([cmin, cmax])
    h2.set_clim([cmin, cmax])
    plt.axis("equal")
    plt.savefig("err_quenched_ball"+str(ntime).zfill(3)+".png")

# %% PLOT Error in L2 norm

err_fphi=[]
err_qappr=[]
angle=[]
for k in range(n_snapshots):
   err_fphi.append( norm(fphi[:,:,k]-qs[:,:,k], 2)/norm(qs[:,:,k],2))
   err_qappr.append(norm(q_appr[:,:,k] - qs[:, :, k], 2)/norm(qs[:,:,k],2))
   angle.append(2*np.pi *k/n_snapshots)
fig = plt.figure(1)
fig.clf()
plt.plot(angle,err_fphi,"r-",label="$\\frac{||f(\\tilde{\phi(x,t)})-q(x,t)||_2}{||q(x,t)||_2}$")
plt.plot(angle,err_qappr,"b-.",label="$\\frac{||\\tilde{q}(x,t)-q(x,t)||_2}{||q(x,t)||_2}$")
plt.xticks([0,np.pi, 2*np.pi],["0","$\pi$","$2\pi$"])
plt.xlabel("$2\pi k/N_{\\mathrm{snapshots}}$")
plt.ylabel("L2-error")
plt.legend()
plt.show()
plt.savefig("L2err_quenched_ball"+str(k)+".png")