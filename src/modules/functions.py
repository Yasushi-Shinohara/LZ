# coding: UTF-8
# This is created 2023/06/18 by Y. Shinohara
# This is lastly modified YYYY/MMD/DD by Y. Shinohara
import sys
from modules.constants import *
import numpy as np
import matplotlib.pyplot as plt

class  LZ:
    """Description for LZ class
    Class for components of Landau-Zener (LZ) problem
    """
    def __init__(self):
        self.var1 = None #A template for the variable

    @classmethod
    def t2h(self, t):
        """Description for t2h function
        A Function to get the LZ Hamiltonian.
        """
        h = np.zeros([2,2], dtype='complex128')
        h[0,0] = t
        h[0,1] = 1.0
        h[1,0] = 1.0
        h[1,1] = -t
        return h
    
    @classmethod
    def h2U(self, h, dt):
        """Description for h2U function
        A Function to get the Unitary matrix for the time-propagation for LZ model.
        """
        e, v = np.linalg.eigh(h)
        U = np.exp(-zI*e[0]*dt)*np.outer(v[:,0], np.conj(v[:,0])) + np.exp(-zI*e[1]*dt)*np.outer(v[:,1], np.conj(v[:,1]))
        return U

    @classmethod
    def solve_LZ(self, tau, epsilon, Nt = 1000, plot_option = True):
        """Description for solve_LZ function
        A Function to get the LZ solution.
        """
        # Setting up time-coordinate
        #Nt = 10000
        ts = -abs(tau)
        te = -1.0*ts # The end and start are setting symmetric from t=0
        t = np.linspace(ts, te, Nt)
        dt = t[1] - t[0]
        nv = np.zeros(Nt, dtype='float64')
        nc = np.zeros(Nt, dtype='float64')
        ncLZ = np.ones(Nt, dtype='float64')*np.exp(-pi/epsilon)
        ev = np.zeros(Nt, dtype='float64')
        ec = np.zeros(Nt, dtype='float64')
        norm = np.zeros(Nt, dtype='float64')
        # Fixing initial condition
        h = self.t2h(ts)/epsilon
        e, v = np.linalg.eigh(h)
        #print(e)
        psi =v[:,0]
        for i in range(Nt):
            h =  self.t2h(t[i])/epsilon
            U = self.h2U(h,dt)
            psi = np.dot(U, psi)
            e, v = np.linalg.eigh(h)
            nv[i] = np.abs(np.vdot(v[:,0], psi))**2
            nc[i] = np.abs(np.vdot(v[:,1], psi))**2
            ev[i] = e[0]
            ec[i] = e[1]
            norm[i] = np.linalg.norm(psi)
        print("nc, ncLZ at the end:", nc[Nt-1], ncLZ[Nt-1])
        
        if (plot_option):
            plt.figure()
            plt.title('The all variables at glance.')
            plt.plot(t, nv, label='nv')
            plt.plot(t, nc, label='nc')
            plt.plot(t,nv+nc, label='nv + nc')
            plt.plot(t,norm, 'k-.', label='norm')
            plt.grid()
            plt.legend()
            plt.show()
            #
            plt.figure()
            plt.title('nc close up')
            plt.plot(t, nc, label='nc')
            plt.plot(t, ncLZ, 'k-.', label='ncLZ')
            plt.grid()
            plt.legend()
            plt.show()

        return t, nv, nc, ncLZ, ev, ec, norm