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

def get_lows(arr, percentile=90):
    """
    Gets the indices of the elements in a 2D NumPy array
    that belong to the bottom `percentile` of the values.

    Args:
    arr: The 2D NumPy array.
    percentile: The percentile of values to consider.

    Returns:
    A tuple of arrays containing the row and column indices
    of the elements in the bottom `percentile`.
    """

    threshold = np.percentile(arr, percentile)  # Calculate the percentile threshold

    # Get the indices of the elements that satisfy the mask
    low_indices = np.where(arr < threshold)
    high_indices = np.where(arr < threshold)

    return low_indices, high_indices

def get_lows(arr, percentile=90):
    """
    Gets the indices of the elements in a 2D NumPy array
    that belong to the bottom `percentile` of the values.

    Args:
    arr: The 2D NumPy array.
    percentile: The percentile of values to consider.

    Returns:
    A tuple of arrays containing the row and column indices
    of the elements in the bottom `percentile`.
    """

    threshold = np.percentile(arr, percentile)  # Calculate the percentile threshold

    # Get the indices of the elements that satisfy the mask
    indices = np.where(arr < threshold)

    return indices




def get_lows_2d(arr, percentile=90):
    """
    Gets the indices of the elements in a 2D NumPy array
    that belong to the bottom `percentile` of the values.

    Args:
    arr: The 2D NumPy array.
    percentile: The percentile of values to consider.

    Returns:
    A tuple of arrays containing the row and column indices
    of the elements in the bottom `percentile`.
    """

    flat_arr = arr.flatten()  # Flatten the array for easier sorting
    threshold = np.percentile(flat_arr, percentile)  # Calculate the percentile threshold

    # Create a boolean mask for values below the threshold
    mask = flat_arr <= threshold

    # Get the indices of the elements that satisfy the mask
    indices = np.where(mask)

    # Convert flat indices to 2D indices
    row_indices, col_indices = np.unravel_index(indices, arr.shape)

    return row_indices, col_indices


 
