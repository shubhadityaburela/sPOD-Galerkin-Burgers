# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 09:11:12 2018

@author: philipp krah
"""

###############################################################################
# IMPORTED MODULES
###############################################################################
import sys
sys.path.append('../lib')
import numpy  as np
from scipy.optimize import minimize
from scipy import interpolate
import scipy.sparse as sp
from numpy.linalg import svd,norm
from numpy import exp, mod,meshgrid
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
###############################################################################
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# %%
def derivative(N, h, coefficient, boundary="periodic"):
    """
    Compute the discrete derivative for periodic BC
    """
    dlow = -(np.size(coefficient)-1)//2
    dup  = - dlow
    
    diagonals = []
    offsets = []
    for k in np.arange(dlow,dup):
        diagonals.append(coefficient[k-dlow]*np.ones(N-abs(k)))
        offsets.append(k)
        if k > 0:
              diagonals.append(coefficient[k-dlow]*np.ones(abs(k)))
              offsets.append(-N+k)
        if k < 0:
              diagonals.append(coefficient[k-dlow]*np.ones(abs(k)))
              offsets.append(N+k)            
        
    return sp.diags(diagonals,offsets)/h

def grad_J2(DX, q0, X0, rank):
    """
    Gradient of the J2 = 1 - sum(sigma)/norm(q,ord='fro') functional
    with respect to spatial variations
    """
    Nshape=np.shape(DX)
    Nx = np.size(q0,0)
    DX = np.reshape(DX,[Nx,-1])
    q = transform(DX,q0,X0)
    stencil = np.asarray([0 , 0,-1, 0, 1,0 , 0])
    #stencil =  np.asarray([ 0, 1, -8, 0,  8,  -1, 0])/12 
    gradx = derivative(Nx, 1, stencil)
    X = X0 + DX
    
    [U,S,Vh] = svd(q, full_matrices=True)
    S = S[:rank]
    U = U[:, :rank]
    Vh = Vh[:rank, :]
    h = np.roll(X,-1,axis=0)-np.roll(X,1,axis=0)
    n2 = np.sum(q**2)
    dqdx = gradx @ (q)
    dqdx = dqdx / h 
    dJdx =  [2* q * dqdx *np.sum(S**2) - 2* n2 *dqdx * np.dot(U * S, Vh) ] / n2**2
    dJdx = np.reshape(dJdx,Nshape)
    return dJdx

def J2_functional(DX,q0,X0,rank):
    """
    J2 = 1 - sum(sigma)/norm(q,ord='fro') functional
    """
    Nx = np.size(q0,0)
    DX = np.reshape(DX,[Nx,-1])
    q = transform(DX,q0,X0)
        
    S = svd(q, compute_uv = False )
    s2 = S[:rank]**2
    n2 = np.sum(q**2)
    J = 1 - sum(s2)/n2
    
    return J

def transform(DX,q0,X0):
    
    Ntime = np.size(q0,1)
    Nx = np.size(q0,0)
    q = np.zeros([Nx,Ntime])
    X = X0 + DX
    #h = x_i+1 - x_i-

    for it in range(Ntime):
       # tck = interpolate.splrep(X0[:,it], q0[:,it], s=0)
       # q[:,it] = interpolate.splev(X[:,it],tck, der=0)
       q[:,it] = np.interp(X[:,it],X0[:,it],q0[:,it],period=X0[-1,it])
    return q

def test_gradient(DX,q0,X0,rank):
    eps=(X0[2,1]-X0[1,1])*0.5
    N=np.shape(q0)
    eps0 = np.zeros(np.shape(DX))
    dJ = np.zeros(np.shape(DX))
    for ix in range(N[0]):
        for it in range(N[1]):
            eps0[ix,it] = eps
            dJ[ix,it] = J2_functional(DX+eps0,q0,X0,rank) -  J2_functional(DX-eps0,q0,X0,rank)
            dJ[ix,it] /= (2*eps)
            eps0[ix,it] = 0
    dJ_fun =  grad_J2(DX,q0,X0,rank)
    err=norm(dJ -dJ_fun)/norm( dJ)
    
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.pcolor(dJ - dJ_fun)
    plt.colorbar()
    plt.title("relative error=" + str(err))
    plt.subplot(1,3,2)
    plt.pcolor(dJ)
    plt.colorbar()
    plt.title("finite diffs")
    plt.subplot(1,3,3)
    plt.pcolor(dJ_fun)
    plt.colorbar()
    plt.title("exact")
    
    plt.savefig("gradient_err_rank"+str(rank)+".png", dpi=300, transparent=True, bbox_inches='tight' )
    return dJ

def plot_result(DX,X0,T0,q0):
    
    plt.figure(figsize=(14,8))
    # plot of the original data
    plt.subplot(1,2,1)
    plt.title('original field: $q(x,t)$\n vector field $(x,t)\mapsto(\Delta(x,t),0)$')
    plt.pcolor(X0,T0,q0)
    plt.colorbar()
    plt.quiver(X0,T0,DX,np.zeros(np.shape(DX)))
    
    plt.xlabel("$x$")
    plt.ylabel("$t$")
    
    # plot of the transformed data
    plt.subplot(1,2,2)
    plt.title('$q(x+\Delta(x,t),t)$')
    phi = transform(DX,q0,X0)
    plt.pcolor(X0,T0,phi)
    plt.colorbar()
    plt.xlabel("$x$")
    plt.ylabel("$t$")    
    plt.savefig("transform.png", dpi=300, transparent=True, bbox_inches='tight' )
    
# %%


N = 100
n_snapshots = 50;
L = 1
T = 2
delta = 0.02    # peak at the midpoint of the soft field
lambd = 0.1    # thickness of the step
R = 0.25        # radius of the cylinder
Nmodes=10
alpha0 = 0.5
x=np.linspace(0,L,N)
t=np.linspace(0,T,n_snapshots)
[T0,X0] = np.meshgrid(t,x)

phis = np.zeros([N, n_snapshots])
qs = np.zeros([N , n_snapshots])

f = lambda x,l : ((np.tanh(x/l) + 1 ) * 0.5)

fig = plt.figure(0)
fig.clf()
ax=fig.gca()
xshift = np.zeros(n_snapshots)
for k in range(n_snapshots):
    z = np.exp(1j*2*np.pi *k/n_snapshots*0.5)
    x0= 0.5*L + R*np.real(z)
    xshift[k]=L*0.5 - x0
    #y0= 0.5 + R*np.imag(z)
    alpha = 0.5#[alpha0*(abs(np.real(z))+0.2)]
    #alpha=1
    phi = np.sqrt(((x-x0)/alpha)**2 + delta**2) -0.15 -delta;
    
    q   = f(phi,lambd); 
    
    phis[:,k] = phi ;    # store distance function 
    qs[:,k] = q   ;    # store field 
    
    if 0:
        fig.clf()
        ax=fig.gca()
        plt.subplot(121)
        plt.plot(x,phi)
        plt.xlim([0,L])
        plt.ylim([min(phi),0.8])
        plt.title("$\phi$")
        plt.ylabel("$x$")
        plt.xlabel("$y$")
        plt.gca().set_aspect('equal', adjustable='box')
        
        
        plt.subplot(122)
        plt.title("$f(\phi)$")
        plt.plot(x,q)
        plt.ylabel("$x$")
        plt.xlabel("$y$")
        plt.gca().set_aspect('equal', adjustable='box')

        #plt.axis("equal")
        #plt.xlim([0,L[0]])
        #plt.ylim([0,L[1]])
        plt.pause(0.01)
        plt.show()
        plt.savefig("1d_shock"+str(k).zfill(3)+".png")
        
# %%
test_gradient(DX0,qs,X0,8)
# %%
DX0 = np.zeros(np.shape(qs))
rank=1
result = minimize(J2_functional, DX0,args=(qs,X0,rank), method='BFGS', jac=grad_J2, options={'disp': True})

# %%
        
        
plt.plot(dJdx[:,6])
plt.plot(qs[:,6])

[Uphi,Sphi,Vphi] = svd(np.reshape(phis,[-1,n_snapshots]), full_matrices=True)
Sphi = Sphi[:Nmodes]
Uphi = Uphi[:, :Nmodes]
Vphi = Vphi[:Nmodes, :]

[Uq,Sq,Vq] = svd(np.reshape(qs,[-1,n_snapshots]),full_matrices=True)
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