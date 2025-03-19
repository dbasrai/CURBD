import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
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

def trial_sem(neural, fs=1, pre_baseline=None, sem = 'neurons'):
    
    #trials x time x neurons
    if sem == 'neurons':
        num_sem = neural.shape[2]
        temp = np.average(neural, axis=0)
        neural_avgsem = np.std(temp, axis=1) / np.sqrt(num_sem) / (fs/1000)
    else:
        num_sem = neural.shape[0]
        temp = np.average(neural, axis=2)
        neural_avgsem = np.std(temp, axis=0) / np.sqrt(num_sem) / (fs/1000)
    neural_avgavg = np.average(neural, axis=(0,2)) / (fs/1000)
    #temp = np.average(neural, axis=2)

    if pre_baseline is not None:
        neural_bs = np.average(neural[:,:pre_baseline,:]) / (fs/1000)
        neural_avgavg = neural_avgavg-neural_bs


    return neural_avgavg, neural_avgsem


def plot_trial_sem(avg, sem, pre=50, colors=None, labels='None', ax=None,
        figsize=None):
    if ax==None:
        fig, ax = plt.subplots(figsize=figsize)
    for idx in range(len(avg)):
        y = avg[idx]
        yhat = sem[idx]
        x = np.arange(len(y)) - pre
        if labels is not None:
            label = labels[idx]
        else:
            label=None
        if colors is not None:
            color = colors[idx]
        else:
            color=None
        ax.plot(x, y, color=color, label=label)
        ax.fill_between(x, y - yhat, y+yhat, color=color, alpha=0.2)
    ax.axvline(0, linestyle='--', color='tab:blue')
    if labels is not 'None':
        ax.legend()
    fig.tight_layout()

    return fig, ax


