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
from numpy import *
from numpy.linalg import svd,norm
import numpy as np
from numpy import exp, mod,meshgrid
import matplotlib.pyplot as plt
import scipy.ndimage.filters as myfilter
from mpl_toolkits import mplot3d
###############################################################################
##################################################################
# Paper for weighted SVD
# https://www.aaai.org/Papers/ICML/2003/ICML03-094.pdf
#Initializing X to q works reasonably well if the weights are bounded
#away from zero, or if the target values in A
#have relatively small variance. However, when the weights are
#zero, or very close to zero, the target values become
#meaningless, and can throw off the search. Initializing
#X
#to zero avoids this problem, as target values with
#zero weights are completely ignored (as they should
#be),    
def weighted_low_rank_svd(A,W, rank=5, Niter=500, max_rel_err=0.01, plot_results=True):
    Xt=np.zeros(shape(W))   
    Nmaxmodes=rank#min(shape(A)) # rank is min(nrows,ncols)
    Nminmodes=rank
    J=[]
    t=0
    err = 1
    while t < Niter and err > max_rel_err:
            Xt = W*A+(1-W)*Xt
            [U,S,V] = svd(Xt,full_matrices=True)
            Nmodes = max(Nminmodes, Nmaxmodes - t)
            S = S[:Nmodes]
            U = U[:, :Nmodes]
            V = V[:Nmodes, :]
            Xt=np.dot(U * S, V)
            err = np.linalg.norm(W*A-W*Xt,2)/norm(W*A,2)
            J.append(err)
            t = t + 1
            print(t, err)
    if plot_results:
        plt.figure
        plt.subplot(1,3,1)
        plt.plot(range(t),J)
        plt.ylabel("weighted error $||\\phi\odot W-\\tilde {\\phi}.W||$")
        plt.subplot(1,3,2)
        plt.title("$\\tilde{\\phi}$")
        plt.pcolor(Xt)
        plt.colorbar()
        plt.subplot(1,3,3)
        plt.pcolor(W*Xt)
        plt.title("$W\\odot \\tilde{\\phi}$")
        plt.colorbar()
    
    return U,S,V
#
def smooth(X,Nwidth):
    #return myfilter.uniform_filter(X,Nwidth, mode="nearest")
    return myfilter.gaussian_filter(X,Nwidth, mode="nearest")