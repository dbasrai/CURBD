import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import copy
import pdb

def bounds2Logical(bounds, duration=None):
    #bounds is a Nx2 array
    if duration is None:
        duration = bounds[-1,1]
    logical = np.zeros(duration)
    for bound in bounds:
        logical[bound[0]:bound[1]] = True
    return logical

def logical2Bounds(logical):
    #messy messy messy
    temp = np.insert(logical, 0, 0)
    new_logical = np.append(temp, 0)
    change = np.diff(new_logical)
    start= change==1
    end= change==-1
    start_idx = np.where(start)[0]
    end_idx = np.where(end)[0]

    
    return (np.vstack((start_idx, end_idx)).T + 1)

def vaf(x,xhat, round_values=True):
    """
    Calculating vaf value
    x: actual values, a numpy array
    xhat: predicted values, a numpy array
    """
    x = x - x.mean(axis=0)
    xhat = xhat - xhat.mean(axis=0)
    if round_values is True:
        return np.round((1-(np.sum(np.square(x -
            xhat))/np.sum(np.square(x)))),2)
    else:
        return (1-(np.sum(np.square(x - xhat))/np.sum(np.square(x))))
def weighted_r2(x, xhat, remove_lows=True):
    score = r2_score(x, xhat, multioutput='raw_values')
    if remove_lows:
        score[score<-1]=-1
    variance = np.var(xhat, axis=0)
    return np.average(score, weights=variance)

def getSeamsFromBounds(bounds, binsize=1):
    #binsize is if you want to downsample
    return np.cumsum(np.diff(bounds, axis=1)//binsize)
 
