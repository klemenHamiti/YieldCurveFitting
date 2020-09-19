# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 22:14:50 2020

@author: kleme
"""

import numpy as np
from scipy.optimize import minimize


class Nelson_Siegel_Svennson(object):
    
    def __init__(self, b=None, tau=None):
        """ Bootstraps yield curve using Nelson-Siegel-Sevensson model 

        keyword arguments:
            b: (np.ndarray) array of beta parameters, dims: (4,)
            tau: (np.ndarray) array of tau parameters: (2,)
        """
        self.params = np.hstack((b, tau))
    
    def _func(self, params, m):
        b, tau = params[:4], params[4:]
        result = b[0]
        result += b[1] * (tau[0] / m) * (1 - np.exp(-m / tau[0]))
        result += b[2] * (tau[0] / m) * (1 - np.exp(-m / tau[0]))
        result -= b[2] * np.exp(-m / tau[0])
        result += b[3] * (tau[1] / m) * (1 - np.exp(-m / tau[1]))
        result -= b[3] * np.exp(-m / tau[1])
        return result
    
    def _residual(self, params, m, yields):
        return np.sum(np.power(yields - self._func(params, m), 2))
    
    def fit(self, m, yields):
        """ Fits the model to the data 
        
        keyword arguments:
            m: (np.ndarray) array of maturities, dims: (n,)
            yields: (np.array) array of bond yields, dims: (n,)
        """
        x = np.ones(6, dtype=np.float64)
        results = minimize(self._residual, x0=x,
                           args=(m, yields), method='L-BFGS-B')
        self.params = results.x
    
    def interpolate(self, m: np.ndarray):
        """ Interploates yield of a bond with a certain maturity, 
        function must be fitted first.
            
        keyword argumetns:
                m: (np.ndarray / float) maturity of a bond
        """
        if type(m) != np.ndarray:
            m = np.array(m)
        try:
            return self._func(self.params, m)
        except IndexError:
            print("Fit the function or provide valid set of hyperparameters")