""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Performs Current-Based Decomposition (CURBD) of multi-region data. Ref:
%
% Perich MG et al. Inferring brain-wide interactions using data-constrained
% recurrent neural network models. bioRxiv. DOI: https://doi.org/10.1101/2020.12.18.423348
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

import math
import random

import numpy as np
import numpy.random as npr
import numpy.linalg

import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
from src.utils.evaluate import *


from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import uniform
from scipy.stats import poisson

from tqdm import tqdm

#from .utils import *

def trainMultiRegionRNN(activity, dtData=1, dtFactor=1, g=1.5, tauRNN=0.01,
                        tauWN=0.1, ampInWN=0.01, nRunTrain=2000,
                        nRunFree=10, P0=1.0,
                        nonLinearity=np.tanh,
                        nonLinearity_inv=np.arctanh,
                        resetPoints=None,
                        plotStatus=True, verbose=True,
                        regions=None):
    r"""
    Trains a data-constrained multi-region RNN. The RNN can be used for,
    among other things, Current-Based Decomposition (CURBD).

    Parameters
    ----------

    activity: numpy.array
        N X T
    dtData: float
        time step (in s) of the training data
    dtFactor: float
        number of interpolation steps for RNN g: float
        instability (chaos); g<1=damped, g>1=chaotic
    tauRNN: float
        decay constant of RNN units
    tauWN: float
        decay constant on filtered white noise inputs
    ampInWN: float
        input amplitude of filtered white noise
    nRunTrain: int
        number of training runs
    nRunFree: int
        number of untrained runs at end
    P0: float
        learning rate
    nonLinearity: function
        inline function for nonLinearity
    resetPoints: list of int
    iemp1 = (temp % 200) > 125
        list of indeces into T. default to only set initial state at time 1.
    plotStatus: bool
        whether to plot data fits during training
    verbose: bool
        whether to print status updates
    regions: dict()
        keys are region names, values are np.array of indeces.
    """
    if dtData is None:
        print('dtData not specified. Defaulting to 1.');
        dtData = 1;
    if resetPoints is None:
        resetPoints = [0, ]
    if regions is None:
        regions = {}

    number_units = activity.shape[0]
    number_learn = activity.shape[0]

    dtRNN = dtData / float(dtFactor)
    nRunTot = nRunTrain + nRunFree

# set up everything for training

    learnList = npr.permutation(number_units)
    iTarget = learnList[:number_learn]
    iNonTarget = learnList[number_learn:]
    tData = dtData*np.arange(activity.shape[1])
    tRNN = np.arange(0, tData[-1] + dtRNN, dtRNN)

    ampWN = math.sqrt(tauWN/dtRNN)
    iWN = ampWN * npr.randn(number_units, len(tRNN))
    inputWN = np.ones((number_units, len(tRNN)))
    for tt in range(1, len(tRNN)):
        inputWN[:, tt] = iWN[:, tt] + (inputWN[:, tt - 1] - iWN[:, tt])*np.exp(- (dtRNN / tauWN))
    inputWN = ampInWN * inputWN


    # initialize directed interaction matrix J
    J = g * npr.randn(number_units, number_units) / math.sqrt(number_units)
    J0 = J.copy()

    # set up target training data
    Adata = activity.copy()
    train_max = Adata.max()
    Adata = Adata/Adata.max()
    Adata = np.minimum(Adata, 0.999)
    Adata = np.maximum(Adata, -0.999)

    # get standard deviation of entire data
    stdData = np.std(Adata[iTarget, :])

    # get indices for each sample of model data
    iModelSample = numpy.zeros(len(tData), dtype=np.int32)
    for i in range(len(tData)):
        iModelSample[i] = (np.abs(tRNN - tData[i])).argmin()

    # initialize some others
    RNN = np.zeros((number_units, len(tRNN)))
    chi2s = []
    pVars = []

    # initialize learning update matrix (see Sussillo and Abbot, 2009)
    PJ = P0*np.eye(number_learn)

    if plotStatus is True:
        plt.rcParams.update({'font.size': 6})
        fig = plt.figure()
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        gs = GridSpec(nrows=2, ncols=4)
    else:
        fig = None

    # start training
    # loop along training runs
    for nRun in range(0, nRunTot):
        H = Adata[:, 0, np.newaxis]
        RNN[:, 0, np.newaxis] = nonLinearity(H)
        # variables to track when to update the J matrix since the RNN and
        # data can have different dt values
        tLearn = 0  # keeps track of current time
        iLearn = 0  # keeps track of last data point learned
        chi2 = 0.0

        for tt in range(1, len(tRNN)):
            # update current learning time
            tLearn += dtRNN
            # check if the current index is a reset point. Typically this won't
            # be used, but it's an option for concatenating multi-trial data
            if tt in resetPoints:
                timepoint = math.floor(tt / dtFactor)
                H = Adata[:, timepoint]
                if H.ndim==1:
                    H = H[:,None]
            # compute next RNN step
            RNN[:, tt, np.newaxis] = nonLinearity(H)
            JR = (J.dot(RNN[:, tt]).reshape((number_units, 1)) +
                  inputWN[:, tt, np.newaxis])
            H = H + dtRNN*(-H + JR)/tauRNN
            # check if the RNN time coincides with a data point to update J
            if tLearn >= dtData:
                tLearn = 0
                err = RNN[:, tt, np.newaxis] - Adata[:, iLearn, np.newaxis]
                iLearn = iLearn + 1
                # update chi2 using this error
                chi2 += np.mean(err ** 2)

                if nRun < nRunTrain:
                    r_slice = RNN[iTarget, tt].reshape(number_learn, 1)
                    k = PJ.dot(r_slice)
                    rPr = (r_slice).T.dot(k)[0, 0]
                    c = 1.0/(1.0 + rPr)
                    PJ = PJ - c*(k.dot(k.T))
                    J[:, iTarget.flatten()] = J[:, iTarget.reshape((number_units))] - c*np.outer(err.flatten(), k.flatten())

        rModelSample = RNN[iTarget, :][:, iModelSample]
        distance = np.linalg.norm(Adata[iTarget, :] - rModelSample)
        pVar = 1 - (distance / (math.sqrt(len(iTarget) * len(tData))
                    * stdData)) ** 2
        pVars.append(pVar)
        chi2s.append(chi2)
        if verbose:
            print('trial=%d pVar=%f chi2=%f' % (nRun, pVar, chi2))
        if fig:
            fig.clear()
            ax = fig.add_subplot(gs[0, 0])
            ax.axis('off')
            ax.imshow(Adata[iTarget, :])
            ax.set_title('real rates')

            ax = fig.add_subplot(gs[0, 1])
            ax.imshow(RNN, aspect='auto')
            ax.set_title('model rates')
            ax.axis('off')

            ax = fig.add_subplot(gs[1, 0])
            ax.plot(pVars)
            ax.set_ylabel('pVar')

            ax = fig.add_subplot(gs[1, 1])
            ax.plot(chi2s)
            ax.set_ylabel('chi2s')

            ax = fig.add_subplot(gs[:, 2:4])
            idx = npr.choice(range(len(iTarget)))
            ax.plot(tRNN, RNN[iTarget[idx], :])
            ax.plot(tData, Adata[iTarget[idx], :])
            ax.set_title(nRun)
            fig.show()
            plt.pause(0.05)

    out_params = {}
    out_params['dtFactor'] = dtFactor
    out_params['number_units'] = number_units
    out_params['g'] = g
    out_params['P0'] = P0
    out_params['tauRNN'] = tauRNN
    out_params['tauWN'] = tauWN
    out_params['ampInWN'] = ampInWN
    out_params['nRunTot'] = nRunTot
    out_params['nRunTrain'] = nRunTrain
    out_params['nRunFree'] = nRunFree
    out_params['nonLinearity'] = nonLinearity
    out_params['resetPoints'] = resetPoints

    out = {}
    out['regions'] = regions
    out['RNN'] = RNN
    out['tRNN'] = tRNN
    out['dtRNN'] = dtRNN
    out['Adata'] = Adata
    out['tData'] = tData
    out['dtData'] = dtData
    out['J'] = J
    out['J0'] = J0
    out['chi2s'] = chi2s
    out['pVars'] = pVars
    out['stdData'] = stdData
    out['inputWN'] = inputWN
    out['iTarget'] = iTarget
    out['iNonTarget'] = iNonTarget
    out['params'] = out_params

    return out

def trainBioMultiRegionRNN(activity, dtData=1, dtFactor=1, g=1.5, tauRNN=0.01,
                        tauWN=0.1, ampInWN=0.01, nRunTrain=2000,
                        nRunFree=10, P0=1.0,
                        nonLinearity=np.tanh,
                        nonLinearity_inv=np.arctanh,
                        resetPoints=None,
                        plotStatus=True, verbose=True,
                        regions=None, g_across=None, 
                        sparse_percent=80,
                        g_loc=(0,0)):
    """
    Trains a data-constrained multi-region RNN. The RNN can be used for,
    among other things, Current-Based Decomposition (CURBD).

    Parameters
    ----------

    activity: numpy.array
        N X T
    dtData: float
        time step (in s) of the training data
    dtFactor: float
        number of interpolation steps for RNN g: float
        instability (chaos); g<1=damped, g>1=chaotic
    tauRNN: float
        decay constant of RNN units
    tauWN: float
        decay constant on filtered white noise inputs
    ampInWN: float
        input amplitude of filtered white noise
    nRunTrain: int
        number of training runs
    nRunFree: int
        number of untrained runs at end
    P0: float
        learning rate
    nonLinearity: function
        inline function for nonLinearity
    resetPoints: list of int
        list of indeces into T. default to only set initial state at time 1.
    plotStatus: bool
        whether to plot data fits during training
    verbose: bool
        whether to print status updates
    regions: dict()
        keys are region names, values are np.array of indeces.
    """
    if dtData is None:
        print('dtData not specified. Defaulting to 1.');
        dtData = 1;
    if resetPoints is None:
        resetPoints = [0, ]
    if regions is None:
        regions = {}
    else:
        num_reg1 = len(regions['region1'])
        num_reg2 = len(regions['region2'])
    if g_across is None:
        g_across = g
    
    number_units = activity.shape[0]
    number_learn = activity.shape[0]

    region1 = regions['region1']
    region2 = regions['region2']

    dtRNN = dtData / float(dtFactor)
    nRunTot = nRunTrain + nRunFree
    percentile_list = np.linspace(sparse_percent//2, sparse_percent, nRunTrain)
    print(percentile_list)
# set up everything for training

    learnList = npr.permutation(number_units)
    iTarget = learnList[:number_learn]
    iNonTarget = learnList[number_learn:]
    tData = dtData*np.arange(activity.shape[1])
    tRNN = np.arange(0, tData[-1] + dtRNN, dtRNN) #true time of RNN

    ampWN = math.sqrt(tauWN/dtRNN)
    iWN = ampWN * npr.randn(number_units, len(tRNN))
    inputWN = np.ones((number_units, len(tRNN)))
    for tt in range(1, len(tRNN)):
        inputWN[:, tt] = iWN[:, tt] + (inputWN[:, tt - 1] - iWN[:, tt])*np.exp(- (dtRNN / tauWN))
    inputWN = ampInWN * inputWN

    # initialize directed interaction matrix J
    #J = g * npr.randn(number_units, number_units) / math.sqrt(number_units)
    J = g * (npr.randn(number_units, number_units))
    J[:num_reg1, :num_reg1] = J[:num_reg1, :num_reg1] + g_loc[0]
    J[num_reg1:, num_reg1:] = J[num_reg1:, num_reg1:] + g_loc[1]
        
    J[:num_reg1, num_reg1:] = (g_across / g) *(truncnorm.rvs(a=0, b=np.inf, 
            loc=0,scale=1,size=(num_reg1, num_reg2)))

    J[num_reg1:, :num_reg1] = (g_across/g)*(truncnorm.rvs(a=0, b=np.inf, 
            loc=0,scale=1,size=(num_reg2, num_reg1)))
    J = J / math.sqrt(number_units)
    J0 = J.copy()

    # set up target training data
    Adata = activity.copy()
    train_max = Adata.max()
    Adata = Adata/Adata.max()
    Adata = np.minimum(Adata, 0.999)
    Adata = np.maximum(Adata, -0.999)

    # get standard deviation of entire data
    stdData = np.std(Adata[iTarget, :])

    # get indices for each sample of model data
    iModelSample = numpy.zeros(len(tData), dtype=np.int32)
    for i in range(len(tData)):
        iModelSample[i] = (np.abs(tRNN - tData[i])).argmin()

    # initialize some others
    RNN = np.zeros((number_units, len(tRNN)))
    chi2s = []
    pVars = []

    # initialize learning update matrix (see Sussillo and Abbot, 2009)
    PJ = P0*np.eye(number_learn)

    if plotStatus is True:
        plt.rcParams.update({'font.size': 6})
        fig = plt.figure()
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        gs = GridSpec(nrows=2, ncols=4)
    else:
        fig = None

    # start training
    # loop along training runs
    for nRun in range(0, nRunTot):
        H = Adata[:, 0, np.newaxis]
        RNN[:, 0, np.newaxis] = nonLinearity(H)
        # variables to track when to update the J matrix since the RNN and
        # data can have different dt values
        tLearn = 0  # keeps track of current time
        iLearn = 0  # keeps track of last data point learned
        chi2 = 0.0

        for tt in range(1, len(tRNN)):
            # update current learning time
            tLearn += dtRNN
            # check if the current index is a reset point. Typically this won't
            # be used, but it's an option for concatenating multi-trial data
            if tt in resetPoints:
                timepoint = math.floor(tt / dtFactor)
                H = Adata[:, timepoint]
                mask = J[:num_reg1, num_reg1:] < 0
                J[:num_reg1, num_reg1:][mask] = 0
                mask = J[num_reg1:, :num_reg1] < 0
                J[num_reg1:, :num_reg1][mask] = 0

                if H.ndim==1:
                    H = H[:,None]
            # compute next RNN step
            RNN[:, tt, np.newaxis] = nonLinearity(H)
            JR = (J.dot(RNN[:, tt]).reshape((number_units, 1)) +
                  inputWN[:, tt, np.newaxis])
            H = H + dtRNN*(-H + JR)/tauRNN
            # check if the RNN time coincides with a data point to update J
            if tLearn >= dtData:
                tLearn = 0
                err = RNN[:, tt, np.newaxis] - Adata[:, iLearn, np.newaxis]
                iLearn = iLearn + 1
                # update chi2 using this error
                chi2 += np.mean(err ** 2)

                if nRun < nRunTrain:
                    r_slice = RNN[iTarget, tt].reshape(number_learn, 1)
                    k = PJ.dot(r_slice)
                    rPr = (r_slice).T.dot(k)[0, 0]
                    c = 1.0/(1.0 + rPr)
                    PJ = PJ - c*(k.dot(k.T))
                    J[:, iTarget.flatten()] = J[:, iTarget.reshape((number_units))] - c*np.outer(err.flatten(), k.flatten())

        if nRun < nRunTrain:
            percentile=percentile_list[nRun]
            print(percentile)
            temp = np.sum(J[:num_reg1, num_reg1:], axis=0) #across region
            low_indices, high_indices = get_lows(temp, percentile=percentile)
            for idx in low_indices:
                J[:num_reg1, num_reg1:][:, idx] = 0
            local_r2 = low_indices

            temp = np.sum(J[num_reg1:, :num_reg1], axis=0)
            low_indices, high_indices = get_lows(temp, percentile=percentile)
            for idx in low_indices:
                J[num_reg1:, :num_reg1][:, idx] = 0

            #adding sparsity, maybe a bad idea

            #temp = np.sum(np.abs(J[:num_reg1, :num_reg1]), axis=0) #across region
            #low_indices, high_indices = get_lows(temp, percentile=percentile)
            #for idx in low_indices:
            #    J[:num_reg1, :num_reg1][:, idx] = 0

            #temp = np.sum(np.abs(J[num_reg1:, num_reg1:]), axis=0) #across region
            #low_indices, high_indices = get_lows(temp, percentile=percentile)
            #for idx in low_indices:
            #    J[num_reg1:, num_reg1:][:, idx] = 0




            
        rModelSample = RNN[iTarget, :][:, iModelSample]
        distance = np.linalg.norm(Adata[iTarget, :] - rModelSample)
        pVar = 1 - (distance / (math.sqrt(len(iTarget) * len(tData))
                    * stdData)) ** 2
        pVars.append(pVar)
        chi2s.append(chi2)
        if verbose:
            print('trial=%d pVar=%f chi2=%f' % (nRun, pVar, chi2))
        if fig:
            fig.clear()
            ax = fig.add_subplot(gs[0, 0])
            ax.axis('off')
            ax.imshow(Adata[iTarget, :])
            ax.set_title('real rates')

            ax = fig.add_subplot(gs[0, 1])
            ax.imshow(RNN, aspect='auto')
            ax.set_title('model rates')
            ax.axis('off')

            ax = fig.add_subplot(gs[1, 0])
            ax.plot(pVars)
            ax.set_ylabel('pVar')
            ax = fig.add_subplot(gs[1, 1])
            ax.plot(chi2s)
            ax.set_ylabel('chi2s')

            ax = fig.add_subplot(gs[:, 2:4])
            idx = npr.choice(range(len(iTarget)))
            ax.plot(tRNN, RNN[iTarget[idx], :])
            ax.plot(tData, Adata[iTarget[idx], :])
            ax.set_title(nRun)
            fig.show()
            plt.pause(0.05)

    out_params = {}
    out_params['dtFactor'] = dtFactor
    out_params['number_units'] = number_units
    out_params['g'] = g
    out_params['P0'] = P0
    out_params['tauRNN'] = tauRNN
    out_params['tauWN'] = tauWN
    out_params['ampInWN'] = ampInWN
    out_params['nRunTot'] = nRunTot
    out_params['nRunTrain'] = nRunTrain
    out_params['nRunFree'] = nRunFree
    out_params['nonLinearity'] = nonLinearity
    out_params['resetPoints'] = resetPoints

    out = {}
    out['regions'] = regions
    out['RNN'] = RNN
    out['tRNN'] = tRNN
    out['dtRNN'] = dtRNN
    out['Adata'] = Adata
    out['tData'] = tData
    out['dtData'] = dtData
    out['J'] = J
    out['J0'] = J0
    out['chi2s'] = chi2s
    out['pVars'] = pVars
    out['stdData'] = stdData
    out['inputWN'] = inputWN
    out['iTarget'] = iTarget
    out['iNonTarget'] = iNonTarget
    out['params'] = out_params
    out['train_max'] = train_max
    out['local_r2']=local_r2

    return out

def trainDaleRNN(activity, dtData=1, dtFactor=1, g=1.5, tauRNN=0.01,
                        tauWN=0.1, ampInWN=0.01, nRunTrain=2000,
                        nRunFree=10, P0=1.0,
                        nonLinearity=np.tanh,
                        nonLinearity_inv=np.arctanh,
                        resetPoints=None,
                        plotStatus=True, verbose=True,
                        regions=None, g_across=None, 
                        sparse_percent=80,
                        g_loc=0):
    """
    Trains a data-constrained multi-region RNN. The RNN can be used for,
    among other things, Current-Based Decomposition (CURBD).

    Parameters
    ----------

    activity: numpy.array
        N X T
    dtData: float
        time step (in s) of the training data
    dtFactor: float
        number of interpolation steps for RNN g: float
        instability (chaos); g<1=damped, g>1=chaotic
    tauRNN: float
        decay constant of RNN units
    tauWN: float
        decay constant on filtered white noise inputs
    ampInWN: float
        input amplitude of filtered white noise
    nRunTrain: int
        number of training runs
    nRunFree: int
        number of untrained runs at end
    P0: float
        learning rate
    nonLinearity: function
        inline function for nonLinearity
    resetPoints: list of int
        list of indeces into T. default to only set initial state at time 1.
    plotStatus: bool
        whether to plot data fits during training
    verbose: bool
        whether to print status updates
    regions: dict()
        keys are region names, values are np.array of indeces.
    """
    if dtData is None:
        print('dtData not specified. Defaulting to 1.');
        dtData = 1;
    if resetPoints is None:
        resetPoints = [0, ]
    if regions is None:
        regions = {}
    else:
        num_reg1 = len(regions['region1'])
        num_reg2 = len(regions['region2'])
    if g_across is None:
        g_across = g
    
    number_units = activity.shape[0]
    number_learn = activity.shape[0]

    region1 = regions['region1']
    region2 = regions['region2']

    dtRNN = dtData / float(dtFactor)
    nRunTot = nRunTrain + nRunFree
    percentile_list = np.linspace(sparse_percent//2, sparse_percent, nRunTrain)
    print(percentile_list)
# set up everything for training

    learnList = npr.permutation(number_units)
    iTarget = learnList[:number_learn]
    iNonTarget = learnList[number_learn:]
    tData = dtData*np.arange(activity.shape[1])
    tRNN = np.arange(0, tData[-1] + dtRNN, dtRNN) #true time of RNN

    ampWN = math.sqrt(tauWN/dtRNN)
    iWN = ampWN * npr.randn(number_units, len(tRNN))
    inputWN = np.ones((number_units, len(tRNN)))
    for tt in range(1, len(tRNN)):
        inputWN[:, tt] = iWN[:, tt] + (inputWN[:, tt - 1] - iWN[:, tt])*np.exp(- (dtRNN / tauWN))
    inputWN = ampInWN * inputWN

    # initialize directed interaction matrix J
    #J = g * npr.randn(number_units, number_units) / math.sqrt(number_units)
    J = g * (npr.randn(number_units, number_units) + g_loc)
        
    J[:num_reg1, num_reg1:] = (g_across / g) *(truncnorm.rvs(a=0, b=np.inf, 
            loc=0,scale=1,size=(num_reg1, num_reg2)))

    J[num_reg1:, :num_reg1] = (g_across/g)*(truncnorm.rvs(a=0, b=np.inf, 
            loc=0,scale=1,size=(num_reg2, num_reg1)))
    J = J / math.sqrt(number_units)
    J0 = J.copy()

    # set up target training data
    Adata = activity.copy()
    train_max = Adata.max()
    Adata = Adata/Adata.max()
    Adata = np.minimum(Adata, 0.999)
    Adata = np.maximum(Adata, -0.999)

    # get standard deviation of entire data
    stdData = np.std(Adata[iTarget, :])

    # get indices for each sample of model data
    iModelSample = numpy.zeros(len(tData), dtype=np.int32)
    for i in range(len(tData)):
        iModelSample[i] = (np.abs(tRNN - tData[i])).argmin()

    # initialize some others
    RNN = np.zeros((number_units, len(tRNN)))
    chi2s = []
    pVars = []

    # initialize learning update matrix (see Sussillo and Abbot, 2009)
    PJ = P0*np.eye(number_learn)

    if plotStatus is True:
        plt.rcParams.update({'font.size': 6})
        fig = plt.figure()
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        gs = GridSpec(nrows=2, ncols=4)
    else:
        fig = None

    # start training
    # loop along training runs
    for nRun in range(0, nRunTot):
        H = Adata[:, 0, np.newaxis]
        RNN[:, 0, np.newaxis] = nonLinearity(H)
        # variables to track when to update the J matrix since the RNN and
        # data can have different dt values
        tLearn = 0  # keeps track of current time
        iLearn = 0  # keeps track of last data point learned
        chi2 = 0.0

        for tt in range(1, len(tRNN)):
            # update current learning time
            tLearn += dtRNN
            # check if the current index is a reset point. Typically this won't
            # be used, but it's an option for concatenating multi-trial data
            if tt in resetPoints:
                timepoint = math.floor(tt / dtFactor)
                H = Adata[:, timepoint]
                mask = J[:num_reg1, num_reg1:] < 0
                J[:num_reg1, num_reg1:][mask] = 0
                mask = J[num_reg1:, :num_reg1] < 0
                J[num_reg1:, :num_reg1][mask] = 0

                if H.ndim==1:
                    H = H[:,None]
            # compute next RNN step
            RNN[:, tt, np.newaxis] = nonLinearity(H)
            JR = (J.dot(RNN[:, tt]).reshape((number_units, 1)) +
                  inputWN[:, tt, np.newaxis])
            H = H + dtRNN*(-H + JR)/tauRNN
            # check if the RNN time coincides with a data point to update J
            if tLearn >= dtData:
                tLearn = 0
                err = RNN[:, tt, np.newaxis] - Adata[:, iLearn, np.newaxis]
                iLearn = iLearn + 1
                # update chi2 using this error
                chi2 += np.mean(err ** 2)

                if nRun < nRunTrain:
                    r_slice = RNN[iTarget, tt].reshape(number_learn, 1)
                    k = PJ.dot(r_slice)
                    rPr = (r_slice).T.dot(k)[0, 0]
                    c = 1.0/(1.0 + rPr)
                    PJ = PJ - c*(k.dot(k.T))
                    J[:, iTarget.flatten()] = J[:, iTarget.reshape((number_units))] - c*np.outer(err.flatten(), k.flatten())

        if nRun < nRunTrain:
            percentile=percentile_list[nRun]
            print(percentile)
            temp = np.sum(J[:num_reg1, num_reg1:], axis=0) #across region
            low_indices, high_indices = get_lows(temp, percentile=percentile)
            J[:num_reg1, num_reg1:][:, low_indices] = 0
            local_r2 = low_indices
            temp = np.sum(J[num_reg1:, local_r2], axis=0)
            low_indices, high_indices = get_lows(temp, percentile=80)
            temp = J[num_reg1:, local_r2][:, low_indices] > 0
            J[num_reg1:, local_r2][:, low_indices][temp] = 0


            temp = np.sum(J[num_reg1:, :num_reg1], axis=0)
            low_indices, high_indices = get_lows(temp, percentile=percentile)
            for idx in low_indices:
                J[num_reg1:, :num_reg1][:, idx] = 0


            
        rModelSample = RNN[iTarget, :][:, iModelSample]
        distance = np.linalg.norm(Adata[iTarget, :] - rModelSample)
        pVar = 1 - (distance / (math.sqrt(len(iTarget) * len(tData))
                    * stdData)) ** 2
        pVars.append(pVar)
        chi2s.append(chi2)
        if verbose:
            print('trial=%d pVar=%f chi2=%f' % (nRun, pVar, chi2))
        if fig:
            fig.clear()
            ax = fig.add_subplot(gs[0, 0])
            ax.axis('off')
            ax.imshow(Adata[iTarget, :])
            ax.set_title('real rates')

            ax = fig.add_subplot(gs[0, 1])
            ax.imshow(RNN, aspect='auto')
            ax.set_title('model rates')
            ax.axis('off')

            ax = fig.add_subplot(gs[1, 0])
            ax.plot(pVars)
            ax.set_ylabel('pVar')
            ax = fig.add_subplot(gs[1, 1])
            ax.plot(chi2s)
            ax.set_ylabel('chi2s')

            ax = fig.add_subplot(gs[:, 2:4])
            idx = npr.choice(range(len(iTarget)))
            ax.plot(tRNN, RNN[iTarget[idx], :])
            ax.plot(tData, Adata[iTarget[idx], :])
            ax.set_title(nRun)
            fig.show()
            plt.pause(0.05)

    out_params = {}
    out_params['dtFactor'] = dtFactor
    out_params['number_units'] = number_units
    out_params['g'] = g
    out_params['P0'] = P0
    out_params['tauRNN'] = tauRNN
    out_params['tauWN'] = tauWN
    out_params['ampInWN'] = ampInWN
    out_params['nRunTot'] = nRunTot
    out_params['nRunTrain'] = nRunTrain
    out_params['nRunFree'] = nRunFree
    out_params['nonLinearity'] = nonLinearity
    out_params['resetPoints'] = resetPoints

    out = {}
    out['regions'] = regions
    out['RNN'] = RNN
    out['tRNN'] = tRNN
    out['dtRNN'] = dtRNN
    out['Adata'] = Adata
    out['tData'] = tData
    out['dtData'] = dtData
    out['J'] = J
    out['J0'] = J0
    out['chi2s'] = chi2s
    out['pVars'] = pVars
    out['stdData'] = stdData
    out['inputWN'] = inputWN
    out['iTarget'] = iTarget
    out['iNonTarget'] = iNonTarget
    out['params'] = out_params
    out['train_max'] = train_max
    out['local_r2']=local_r2

    return out



def trainLaserMultiRegionRNN(activity, pre, post, dtData=1, dtFactor=1, g=1.5, tauRNN=0.01,
                        tauWN=0.1, ampInWN=0.01, nRunTrain=2000,
                        nRunFree=10, P0=1.0,
                        nonLinearity=np.tanh,
                        nonLinearity_inv=np.arctanh,
                        resetPoints=None,
                        plotStatus=True, verbose=True,
                        regions=None, g_across=None,
                        corrnoise=True, optoAmp=.01):
    """
    Trains a data-constrained multi-region RNN. The RNN can be used for,
    among other things, Current-Based Decomposition (CURBD).

    Parameters
    ----------

    activity: numpy.array
        N X T
    dtData: float
        time step (in s) of the training data
    dtFactor: float
        number of interpolation steps for RNN g: float
        instability (chaos); g<1=damped, g>1=chaotic
    tauRNN: float
        decay constant of RNN units
    tauWN: float
        decay constant on filtered white noise inputs
    ampInWN: float
        input amplitude of filtered white noise
    nRunTrain: int
        number of training runs
    nRunFree: int
        number of untrained runs at end
    P0: float
        learning rate
    nonLinearity: function
        inline function for nonLinearity
    resetPoints: list of int
        list of indeces into T. default to only set initial state at time 1.
    plotStatus: bool
        whether to plot data fits during training
    verbose: bool
        whether to print status updates
    regions: dict()
        keys are region names, values are np.array of indeces.
    """
    if dtData is None:
        print('dtData not specified. Defaulting to 1.');
        dtData = 1;
    if resetPoints is None:
        resetPoints = [0, ]
    if regions is None:
        regions = {}
    else:
        num_reg1 = len(regions['region1'])
        num_reg2 = len(regions['region2'])
    if g_across is None:
        g_across = g

    number_units = activity.shape[0]
    number_learn = activity.shape[0]

    region1 = regions['region1']
    region2 = regions['region2']

    dtRNN = dtData / float(dtFactor)
    nRunTot = nRunTrain + nRunFree

# set up everything for training

    learnList = npr.permutation(number_units)
    iTarget = learnList[:number_learn]
    iNonTarget = learnList[number_learn:]
    tData = dtData*np.arange(activity.shape[1])
    tRNN = np.arange(0, tData[-1] + dtRNN, dtRNN) #true time of RNN

    ampWN = math.sqrt(tauWN/dtRNN)
    if corrnoise:
        meanz = np.zeros(number_units) 
        covz = np.zeros((number_units, number_units))
        d = number_units    # number of dimensions
        k = 5# number of factors

        W = np.random.randn(d,k)
        S = W@W.T + np.diag(np.random.rand(1,d))
        S =np.diag(1/np.sqrt(np.diag(S))) @ S @ np.diag(1/np.sqrt(np.diag(S)))

        #covz[:num_r1, :num_r1] = S
        covz = S
        covz[:num_reg1, num_reg1:] = 0
        covz[num_reg1:, :num_reg1] = 0
        np.fill_diagonal(covz, 1)
        #ampWN=1

        iWN  = ampWN * mvn.rvs(mean=meanz, cov=covz, size=(len(tRNN))).T
    else:

        iWN = ampWN * npr.randn(number_units, len(tRNN))
    optoInp = np.zeros((number_units, len(tRNN)))
    trial_max = pre+post+25
    temp = np.arange(len(tRNN))
    temp1 = (temp % trial_max) > pre
    temp2 = (temp % trial_max) < pre+25
    temp3 = temp1 & temp2
    mask = np.arange(len(tRNN))[temp3]
    for i in mask:
        print(i)
        optoInp[region2, i] = truncnorm.rvs(a=-np.inf, b=0, loc=0, scale=1,
               size=len(region2))


    inputWN = np.ones((number_units, len(tRNN)))
    for tt in range(1, len(tRNN)):
        inputWN[:, tt] = iWN[:, tt] + (inputWN[:, tt - 1] - iWN[:, tt])*np.exp(- (dtRNN / tauWN))
    inputWN = ampInWN * inputWN
    optoInp = optoAmp * optoInp

    # initialize directed interaction matrix J
    J = g * ((npr.randn(number_units, number_units))-.2)
        
    J[:num_reg1, num_reg1:] = (g_across / g) *truncnorm.rvs(a=0, b=np.inf, 
            loc=0,scale=1,size=(num_reg1, num_reg2))

    J[num_reg1:, :num_reg1] = (g_across/g)*truncnorm.rvs(a=0, b=np.inf, 
            loc=0,scale=1,size=(num_reg2, num_reg1))
    J = J / math.sqrt(number_units)
    J0 = J.copy()

    # set up target training data
    Adata = activity.copy()
    train_max = Adata.max()
    Adata = Adata/Adata.max()
    Adata = np.minimum(Adata, 0.999)
    Adata = np.maximum(Adata, -0.999)

    # get standard deviation of entire data
    stdData = np.std(Adata[iTarget, :])

    # get indices for each sample of model data
    iModelSample = numpy.zeros(len(tData), dtype=np.int32)
    for i in range(len(tData)):
        iModelSample[i] = (np.abs(tRNN - tData[i])).argmin()

    # initialize some others
    RNN = np.zeros((number_units, len(tRNN)))
    chi2s = []
    pVars = []

    # initialize learning update matrix (see Sussillo and Abbot, 2009)
    PJ = P0*np.eye(number_learn)

    if plotStatus is True:
        plt.rcParams.update({'font.size': 6})
        fig = plt.figure()
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        gs = GridSpec(nrows=2, ncols=4)
    else:
        fig = None

    # start training
    # loop along training runs
    for nRun in range(0, nRunTot):
        H = Adata[:, 0, np.newaxis]
        RNN[:, 0, np.newaxis] = nonLinearity(H)
        # variables to track when to update the J matrix since the RNN and
        # data can have different dt values
        tLearn = 0  # keeps track of current time
        iLearn = 0  # keeps track of last data point learned
        chi2 = 0.0

        for tt in range(1, len(tRNN)):
            # update current learning time
            tLearn += dtRNN
            # check if the current index is a reset point. Typically this won't
            # be used, but it's an option for concatenating multi-trial data
            if tt in resetPoints:
                timepoint = math.floor(tt / dtFactor)
                H = Adata[:, timepoint]
                 
                mask = J[:num_reg1, num_reg1:] < 0
                J[:num_reg1, num_reg1:][mask] =0
                mask = J[num_reg1:, :num_reg1] < 0
                J[num_reg1:, :num_reg1][mask] = 0

                if H.ndim==1:
                    H = H[:,None]
            # compute next RNN step
            RNN[:, tt, np.newaxis] = nonLinearity(H)
            JR = (J.dot(RNN[:, tt]).reshape((number_units, 1)) +
                    inputWN[:, tt, np.newaxis]) + optoInp[:,tt, np.newaxis]
            H = H + dtRNN*(-H + JR)/tauRNN
            # check if the RNN time coincides with a data point to update J
            if tLearn >= dtData:
                tLearn = 0
                err = RNN[:, tt, np.newaxis] - Adata[:, iLearn, np.newaxis]
                iLearn = iLearn + 1
                # update chi2 using this error
                chi2 += np.mean(err ** 2)

                if nRun < nRunTrain:
                    r_slice = RNN[iTarget, tt].reshape(number_learn, 1)
                    k = PJ.dot(r_slice)
                    rPr = (r_slice).T.dot(k)[0, 0]
                    c = 1.0/(1.0 + rPr)
                    PJ = PJ - c*(k.dot(k.T))
                    J[:, iTarget.flatten()] = J[:, iTarget.reshape((number_units))] - c*np.outer(err.flatten(), k.flatten())


        rModelSample = RNN[iTarget, :][:, iModelSample]
        distance = np.linalg.norm(Adata[iTarget, :] - rModelSample)
        pVar = 1 - (distance / (math.sqrt(len(iTarget) * len(tData))
                    * stdData)) ** 2
        pVars.append(pVar)
        chi2s.append(chi2)
        #masky = J[:num_reg1, num_reg1:] < 0.01
        #print(np.count_nonzero(masky))
 
        if verbose:
            print('trial=%d pVar=%f chi2=%f' % (nRun, pVar, chi2))
        if fig:
            fig.clear()
            ax = fig.add_subplot(gs[0, 0])
            ax.axis('off')
            ax.imshow(Adata[iTarget, :])
            ax.set_title('real rates')

            ax = fig.add_subplot(gs[0, 1])
            ax.imshow(RNN, aspect='auto')
            ax.set_title('model rates')
            ax.axis('off')

            ax = fig.add_subplot(gs[1, 0])
            ax.plot(pVars)
            ax.set_ylabel('pVar')
            ax = fig.add_subplot(gs[1, 1])
            ax.plot(chi2s)
            ax.set_ylabel('chi2s')

            ax = fig.add_subplot(gs[:, 2:4])
            idx = npr.choice(range(len(iTarget)))
            ax.plot(tRNN, RNN[iTarget[idx], :])
            ax.plot(tData, Adata[iTarget[idx], :])
            ax.set_title(nRun)
            fig.show()
            plt.pause(0.05)

    out_params = {}
    out_params['dtFactor'] = dtFactor
    out_params['number_units'] = number_units
    out_params['g'] = g
    out_params['P0'] = P0
    out_params['tauRNN'] = tauRNN
    out_params['tauWN'] = tauWN
    out_params['ampInWN'] = ampInWN
    out_params['nRunTot'] = nRunTot
    out_params['nRunTrain'] = nRunTrain
    out_params['nRunFree'] = nRunFree
    out_params['nonLinearity'] = nonLinearity
    out_params['resetPoints'] = resetPoints

    out = {}
    out['regions'] = regions
    out['RNN'] = RNN
    out['tRNN'] = tRNN
    out['dtRNN'] = dtRNN
    out['Adata'] = Adata
    out['tData'] = tData
    out['dtData'] = dtData
    out['J'] = J
    out['J0'] = J0
    out['chi2s'] = chi2s
    out['pVars'] = pVars
    out['stdData'] = stdData
    out['inputWN'] = inputWN
    out['iTarget'] = iTarget
    out['iNonTarget'] = iNonTarget
    out['params'] = out_params
    out['train_max'] = train_max

    return out


def simulate_plus_optoinput(model,t,wn_t, tauRNN=None, ampInWN=None,
        tauWN=None, optoAmp=None, corrnoise=True, sparse=None,
        inhib_only=False):
    assert np.max(wn_t) <= t, print('issue')
    dtRNN = model['dtRNN']
    params = model['params']
    if tauWN is None:
        tauWN = params['tauWN']
    if tauRNN is None:
        tauRNN = params['tauRNN']
    number_units = params['number_units']
    if ampInWN is None:
        ampInWN = params['ampInWN']
    nonLinearity = params['nonLinearity']
    J = model['J']
    Adata = model['Adata']
    region1 = model['regions']['region1']
    region2 = model['regions']['region2']

    r1 = model['regions']['region1']
    r2 = model['regions']['region2']
    num_r1 = len(r1)
    num_r2 = len(r2)
    if optoAmp is None:
        optoAmp = ampInWN
        

    stab_t = .1*t
    dtStab = int(stab_t / dtRNN)
    wn_t = wn_t + stab_t
    dt_wnt = wn_t / dtRNN
    dt_wnt = dt_wnt.astype(int)

    tRNN = np.arange(0, t+stab_t, dtRNN)
    wn_t_logical = bounds2Logical(dt_wnt, duration=int(tRNN[-1]/dtRNN)+1)

    ampWN = math.sqrt(tauWN/dtRNN)
    if ampWN <1:
        ampWN=1

    ampWN=1
    if corrnoise:
        meanz = np.zeros(number_units) 
        covz = np.zeros((number_units, number_units))
        d = number_units    # number of dimensions
        k = 5# number of factors

        W = np.random.randn(d,k)
        S = W@W.T + np.diag(np.random.rand(1,d))
        S =np.diag(1/np.sqrt(np.diag(S))) @ S @ np.diag(1/np.sqrt(np.diag(S)))



        #covz[:num_r1, :num_r1] = S
        covz = S
        np.fill_diagonal(covz, 1)
        ampWN=1

        iWN  = ampWN * (mvn.rvs(mean=meanz, cov=covz, size=(len(tRNN))).T)
    else:
        iWN = npr.randn(number_units, len(tRNN))

    inputWN = np.ones((number_units, len(tRNN)))
    wn_idx = np.arange(len(tRNN))[wn_t_logical.astype(bool)]#horrific
    
    for tt in range(1, len(tRNN)):
        if tauWN==0:
            inputWN[:,tt] = iWN[:,tt]
        else:
            inputWN[:, tt] = iWN[:, tt] + (inputWN[:, tt - 1] - iWN[:, tt])*np.exp(- (dtRNN / tauWN))
    
    if inhib_only:
        r2r1 = model['J'][:num_r1, num_r1:]
        inhibs = np.sum(r2r1, axis=0) < 5
        rest = numpy.full(num_r1, False)
        inhibs = np.concatenate((rest, inhibs))
        num_inhibs=np.count_nonzero(inhibs)
        print(f'num_inhibs:{num_inhibs}')

    optoInp =  np.zeros((number_units, len(tRNN))) 
    for idx, i in enumerate(wn_idx):
        
        if inhib_only:
            optoInp[inhibs, i] = truncnorm.rvs(a=0, b=np.inf, loc=0,
                   scale=1,size=num_inhibs) 
        elif sparse is None:
            optoInp[region2, i] = truncnorm.rvs(a=-np.inf, b=0, loc=0,
                   scale=1,size=num_r2) 
        else:
            indices = model['local_r2'][0]
            #sparse_indices = np.random.choice(indices, size=len(indices)//4,
                   # replace=False)
            localz = region2[indices] 
            optoInp[localz, i] = truncnorm.rvs(a=0, b=np.inf, loc=0,
                   scale=1,size=len(localz)) 



    inputWN = ampInWN * inputWN
    optoInp = optoAmp * optoInp

    #output simulation

    stabilize = int(.1 * len(tRNN))
    sim = np.zeros((number_units, len(tRNN))) 

    #randomly initialize based on data
    H = Adata[:, np.random.choice(len(Adata))]
    if H.ndim==1:
        sim[:,0] = nonLinearity(H)
    else:
        sim[:, 0, np.newaxis] = nonLinearity(H)


    for tt in tqdm(range(1, len(tRNN))):
        # check if the current index is a reset point. Typically this won't
        # be used, but it's an option for concatenating multi-trial data
        # computoe next RNN step
            
        if H.ndim==1:
            sim[:,tt] = nonLinearity(H)
        else:
            sim[:, tt, np.newaxis] = nonLinearity(H)
        
        #sim[:, tt, np.newaxis] = nonLinearity(H)
        JR = (J.dot(sim[:, tt]).reshape((number_units, 1)) +
                inputWN[:, tt, np.newaxis]) + optoInp[:, tt, np.newaxis]
        JR = np.squeeze(JR)
        H = H + dtRNN*(-H + JR)/tauRNN

    return sim[:,dtStab:], wn_t_logical[dtStab:]

def simulate_x_optoinput(model,t,wn_t, tauRNN=None, ampInWN=None,
        tauWN=None, opto_loc=0):
    assert np.max(wn_t) <= t, print('issue')
    dtRNN = model['dtRNN']
    params = model['params']
    if tauWN is None:
        tauWN = params['tauWN']
    if tauRNN is None:
        tauRNN = params['tauRNN']
    number_units = params['number_units']
    if ampInWN is None:
        ampInWN = params['ampInWN']
    nonLinearity = params['nonLinearity']
    J = model['J']
    Adata = model['Adata']
    region1 = model['regions']['region1']
    region2 = model['regions']['region2']

    r1 = model['regions']['region1']
    r2 = model['regions']['region2']
    num_r1 = len(r1)
    num_r2 = len(r2)
        

    stab_t = .1*t
    dtStab = int(stab_t / dtRNN)
    wn_t = wn_t + stab_t

    dt_wnt = wn_t / dtRNN
    dt_wnt = dt_wnt.astype(int)

    tRNN = np.arange(0, t+stab_t, dtRNN)
    wn_t_logical = bounds2Logical(dt_wnt, duration=int(tRNN[-1]/dtRNN)+1)

    iWN = npr.randn(number_units, len(tRNN))


    inputWN = np.ones((number_units, len(tRNN)))
    wn_idx = np.arange(len(tRNN))[wn_t_logical.astype(bool)]#horrific
    for idx, i in enumerate(wn_idx):

       iWN[region2, i] = truncnorm.rvs(a=-np.inf, b=0, loc=0, scale=.1,
               size=len(region2))
       #iWN[region2, i]=npr.randn(len(region2))-.1

    
    for tt in range(1, len(tRNN)):
        if tauWN==0:
            inputWN[:,tt] = iWN[:,tt]
        else:
            inputWN[:, tt] = iWN[:, tt] + (inputWN[:, tt - 1] - iWN[:, tt])*np.exp(- (dtRNN / tauWN))

    ampInWN = 1
    inputWN = ampInWN * inputWN
    stabilize = int(.1 * len(tRNN))
    sim = np.zeros((number_units, len(tRNN))) 

    #randomly initialize based on data
    H = Adata[:, np.random.choice(len(Adata))]
    if H.ndim==1:
        sim[:,0] = nonLinearity(H)
    else:
        sim[:, 0, np.newaxis] = nonLinearity(H)


    for tt in tqdm(range(1, len(tRNN))):
        # check if the current index is a reset point. Typically this won't
        # be used, but it's an option for concatenating multi-trial data
        # computoe next RNN step
            
        if H.ndim==1:
            sim[:,tt] = nonLinearity(H)
        else:
            sim[:, tt, np.newaxis] = nonLinearity(H)
        
        #sim[:, tt, np.newaxis] = nonLinearity(H)
        JR = (J.dot(sim[:, tt]).reshape((number_units, 1)) +
                inputWN[:, tt, np.newaxis])
        JR = np.squeeze(JR)
        H = H + dtRNN*(-H + JR)/tauRNN

    return sim[:,dtStab:]



def simulate_corrnoise(model, t, tauRNN=None, ampInWN=None, tauWN=None):
    #randomly initialize from initial condition of training data
    dtRNN = model['dtRNN']
    params = model['params']
    if tauWN is None:
        tauWN = params['tauWN']
    if tauRNN is None:
        tauRNN = params['tauRNN']
    number_units = params['number_units']
    if ampInWN is None:
        ampInWN = params['ampInWN']
    nonLinearity = params['nonLinearity']
    J = model['J']
    Adata = model['Adata']
    r1 = model['regions']['region1']
    r2 = model['regions']['region2']
    num_reg1 = len(r1)
    num_reg2 = len(r2)

    stab_t = .1*t
    dtStab = int(stab_t / dtRNN)

    tRNN = np.arange(0, t+stab_t, dtRNN)
    ampWN = math.sqrt(tauWN/dtRNN)
    if ampWN < 1:
        ampWN =1

    meanz = np.zeros(number_units) 
    covz = np.zeros((number_units, number_units))
    d = number_units    # number of dimensions
    k = 5# number of factors

    W = np.random.randn(d,k)
    S = W@W.T + np.diag(np.random.rand(1,d))
    S =np.diag(1/np.sqrt(np.diag(S))) @ S @ np.diag(1/np.sqrt(np.diag(S)))



    #covz[:num_r1, :num_r1] = S
    covz=S
    covz[:num_reg1, num_reg1:] = 0
    covz[num_reg1:, :num_reg1] = 0
    np.fill_diagonal(covz, 1)
    ampWN=1

    iWN  = ampWN * (mvn.rvs(mean=meanz, cov=covz, size=(len(tRNN))).T)
    inputWN = np.ones((number_units, len(tRNN)))
    for tt in range(1, len(tRNN)):
        if tauWN == 0:
            inputWN[:, tt] = iWN[:, tt] 
        else:
            inputWN[:, tt] = iWN[:, tt] + (inputWN[:, tt - 1] - iWN[:, tt])*np.exp(- (dtRNN / tauWN))
    inputWN = ampInWN * inputWN
    
    #output simulation
    sim = np.zeros((number_units, len(tRNN))) 

    #randomly initialize based on data
    H = Adata[:, np.random.choice(len(Adata))]
    if H.ndim==1:
        sim[:,0] = nonLinearity(H)
    else:
        sim[:, 0, np.newaxis] = nonLinearity(H)
    
    wn_contrib=[]
    j_contrib=[]

    for tt in tqdm(range(1, len(tRNN))):
        # check if the current index is a reset point. Typically this won't
        # be used, but it's an option for concatenating multi-trial data
        # computoe next RNN step
        if H.ndim==1:
            sim[:,tt] = nonLinearity(H)
        else:
            sim[:, tt, np.newaxis] = nonLinearity(H)
        
        #sim[:, tt, np.newaxis] = nonLinearity(H)
        #wn_contrib.append(np.sum(np.abs(inputWN[:,tt])))
        #j_contrib.append(np.sum(np.abs(J.dot(sim[:, tt]))))
        JR = (J.dot(sim[:, tt]).reshape((number_units, 1)) +
              inputWN[:, tt, np.newaxis])
        JR = np.squeeze(JR)
        H = H + dtRNN*(-H + JR)/tauRNN

    return sim[:,dtStab:]





def simulate(model, t, tauRNN=None, ampInWN=None, tauWN=None):
    #randomly initialize from initial condition of training data
    dtRNN = model['dtRNN']
    params = model['params']
    if tauWN is None:
        tauWN = params['tauWN']
    if tauRNN is None:
        tauRNN = params['tauRNN']
    number_units = params['number_units']
    if ampInWN is None:
        ampInWN = params['ampInWN']
    nonLinearity = params['nonLinearity']
    J = model['J']
    Adata = model['Adata']

    stab_t = .1*t
    dtStab = int(stab_t / dtRNN)

    tRNN = np.arange(0, t+stab_t, dtRNN)
    ampWN = math.sqrt(tauWN/dtRNN)
    if ampWN < 1:
        ampWN =1
    ampWN = 1
    iWN = ampWN * npr.randn(number_units, len(tRNN))
    inputWN = np.ones((number_units, len(tRNN)))
    for tt in range(1, len(tRNN)):
        if tauWN==0:
            inputWN[:, tt] = iWN[:, tt] 
        else:
            inputWN[:, tt] = iWN[:, tt] + (inputWN[:, tt - 1] - iWN[:, tt])*np.exp(- (dtRNN / tauWN))
    inputWN = ampInWN * inputWN
    
    #output simulation

    sim = np.zeros((number_units, len(tRNN))) 

    #randomly initialize based on data
    H = Adata[:, np.random.choice(len(Adata))]
    if H.ndim==1:
        sim[:,0] = nonLinearity(H)
    else:
        sim[:, 0, np.newaxis] = nonLinearity(H)

    for tt in tqdm(range(1, len(tRNN))):
        # check if the current index is a reset point. Typically this won't
        # be used, but it's an option for concatenating multi-trial data
        # computoe next RNN step
        if H.ndim==1:
            sim[:,tt] = nonLinearity(H)
        else:
            sim[:, tt, np.newaxis] = nonLinearity(H)
        
        #sim[:, tt, np.newaxis] = nonLinearity(H)
        JR = (J.dot(sim[:, tt]).reshape((number_units, 1)) +
              inputWN[:, tt, np.newaxis])
        JR = np.squeeze(JR)
        H = H + dtRNN*(-H + JR)/tauRNN

    return sim[:,dtStab:]


def plotFit(model):
    pVars = model['pVars']
    RNN = model['RNN']
    tRNN = model['tRNN']
    dtFactor = model['params']['dtFactor']
    Adata=model['Adata']
    iTarget = model['iTarget']
    tData = model['tData']
    chi2s = model['chi2s']

    r2 = weighted_r2(Adata.T, RNN[:, ::dtFactor].T)
    print(r2)


    #plt.rcParams.update({'font.size': 6})
    fig = plt.figure()
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    gs = GridSpec(nrows=2, ncols=4)

    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    ax.imshow(Adata, aspect='auto')
    ax.set_title('real rates')

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(RNN, aspect='auto')
    ax.set_title('model rates')
    ax.axis('off')

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(pVars)
    ax.set_ylabel('pVar')

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(chi2s)
    ax.set_ylabel('chi2s')

    ax = fig.add_subplot(gs[:, 2:4])
    idx = npr.choice(range(len(iTarget)))
    ax.plot(tData, Adata[iTarget[idx], :], label='true')
    ax.plot(tRNN, RNN[iTarget[idx], :], label='predicted')
    ax.set_title(f'weighted_r2:{r2:.2f}')
    ax.legend()
    plt.pause(0.05)




def threeRegionSim(number_units=100,
                   ga=1.8,
                   gb=1.5,
                   gc=1.5,
                   tau=0.1,
                   fracInterReg=0.05,
                   ampInterReg=0.02,
                   fracExternal=0.5,
                   ampInB=1,
                   ampInC=-1,
                   dtData=0.01,
                   T=10,
                   leadTime=2,
                   bumpStd=0.2,
                   plotSim=True):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % out = threeRegionSim(params)
    %
    % Generates a simulated dataset with three interacting regions. Ref:
    %
    % Perich MG et al. Inferring brain-wide interactions using data-constrained
    % recurrent neural network models. bioRxiv. DOI:
    %
    % INPUTS:
    %   params : (optional) parameter struct. See code below for options.
    %
    % OUTPUTS:
    %   out : output struct with simulation results and parameters
    %
    % Written by Matthew G. Perich. Updated December 2020.
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Parameters
    ----------

    number_units: int
        number of units in each region
    ga: float
        chaos parameter for Region A
    gb: float
        chaos parameter for Region B
    gc: float
        chaos parameter for Region C
    tau: float
        decay time constant of RNNs
    fracInterReg: float
        fraction of inter-region connections
    ampInterReg: float
        amplitude of inter-region connections
    fracExternal: float
        fraction of external inputs to B/C
    ampInB: float
        amplitude of external inputs to Region B
    ampInC: float
        amplitude of external inputs to Region C
    dtData: float
        time step (s) of the simulation
    T: float
        total simulation time
    leadTime: float
        time before sequence starts and after FP moves
    bumpStd: float
        width (in frac of population) of sequence/FP
    plotSim: bool
        whether to plot the results
    """
    tData = np.arange(0, (T + dtData), dtData)

    # for now it only works if the networks are the same size
    Na = Nb = Nc = number_units

    # set up RNN A (chaotic responder)
    Ja = npr.randn(Na, Na)
    Ja = ga / math.sqrt(Na) * Ja
    hCa = 2 * npr.rand(Na, 1) - 1  # start from random state

    # set up RNN B (driven by sequence)
    Jb = npr.randn(Nb, Nb)
    Jb = gb / math.sqrt(Na) * Jb
    hCb = 2 * npr.rand(Nb, 1) - 1  # start from random state

    # set up RNN C (driven by fixed point)
    Jc = npr.randn(Nc, Nc)
    Jc = gb / math.sqrt(Na) * Jc
    hCc = 2 * npr.rand(Nc, 1) - 1  # start from random state

    # generate external inputs
    # set up sequence-driving network
    xBump = np.zeros((Nb, len(tData)))
    sig = bumpStd*Nb  # width of bump in N units

    norm_by = 2*sig ** 2
    cut_off = math.ceil(len(tData)/2) - 100
    for i in range(Nb):
        stuff = (i - sig - Nb * tData / (tData[-1] / 2)) ** 2 / norm_by
        xBump[i, :] = np.exp(-stuff)
        xBump[i, cut_off:] = xBump[i, cut_off]

    hBump = np.log((xBump+0.01)/(1-xBump+0.01))
    hBump = hBump-np.min(hBump)
    hBump = hBump/np.max(hBump)

    # set up fixed points driving network

    xFP = np.zeros((Nc, len(tData)))
    cut_off = math.ceil(len(tData)/2) + 100
    for i in range(Nc):
        front = xBump[i, 10] * np.ones((1, cut_off))
        back = xBump[i, 300] * np.ones((1, len(tData)-cut_off))
        xFP[i, :] = np.concatenate((front, back), axis=1)
    hFP = np.log((xFP+0.01)/(1-xFP+0.01))
    hFP = hFP - np.min(hFP)
    hFP = hFP/np.max(hFP)

    # add the lead time
    extratData = np.arange(tData[-1] + dtData, T + leadTime, dtData)
    tData = np.concatenate((tData, extratData))

    newmat = np.tile(hBump[:, 1, np.newaxis], (1, math.ceil(leadTime/dtData)))
    hBump = np.concatenate((newmat, hBump), axis=1)

    newmat = np.tile(hFP[:, 1, np.newaxis], (1, math.ceil(leadTime/dtData)))
    hFP = np.concatenate((newmat, hFP), axis=1)

    # build connectivity between RNNs
    Nfrac = int(fracInterReg*number_units)

    rand_idx = npr.permutation(number_units)
    w_A2B = np.zeros((number_units, 1))
    w_A2B[rand_idx[0:Nfrac]] = 1

    rand_idx = npr.permutation(number_units)
    w_A2C = np.zeros((number_units, 1))
    w_A2C[rand_idx[0:Nfrac]] = 1

    rand_idx = npr.permutation(number_units)
    w_B2A = np.zeros((number_units, 1))
    w_B2A[rand_idx[0:Nfrac]] = 1

    rand_idx = npr.permutation(number_units)
    w_B2C = np.zeros((number_units, 1))
    w_B2C[rand_idx[0:Nfrac]] = 1

    rand_idx = npr.permutation(number_units)
    w_C2A = np.zeros((number_units, 1))
    w_C2A[rand_idx[0:Nfrac]] = 1

    rand_idx = npr.permutation(number_units)
    w_C2B = np.zeros((number_units, 1))
    w_C2B[rand_idx[0:Nfrac]] = 1

    # Sequence only projects to B
    Nfrac = int(fracExternal * number_units)
    rand_idx = npr.permutation(number_units)
    w_Seq2B = np.zeros((number_units, 1))
    w_Seq2B[rand_idx[0:Nfrac]] = 1

    # Fixed point only projects to A
    Nfrac = int(fracExternal * number_units)
    rand_idx = npr.permutation(number_units)
    w_Fix2C = np.zeros((number_units, 1))
    w_Fix2C[rand_idx[0:Nfrac]] = 1

    # generate time series simulated data
    Ra = np.empty((Na, len(tData)))
    Ra[:] = np.NaN

    Rb = np.empty((Nb, len(tData)))
    Rb[:] = np.NaN

    Rc = np.empty((Nc, len(tData)))
    Rc[:] = np.NaN

    for tt in range(len(tData)):
        Ra[:, tt, np.newaxis] = np.tanh(hCa)
        Rb[:, tt, np.newaxis] = np.tanh(hCb)
        Rc[:, tt, np.newaxis] = np.tanh(hCc)
        # chaotic responder
        JRa = Ja.dot(Ra[:, tt, np.newaxis])
        JRa += ampInterReg * w_B2A * Rb[:, tt, np.newaxis]
        JRa += ampInterReg * w_C2A * Rc[:, tt, np.newaxis]
        hCa = hCa + dtData * (-hCa + JRa) / tau

        # sequence driven
        JRb = Jb.dot(Rb[:, tt, np.newaxis])
        JRb += ampInterReg * w_A2B * Ra[:, tt, np.newaxis]
        JRb += ampInterReg * w_C2B * Rc[:, tt, np.newaxis]
        JRb += ampInB * w_Seq2B * hBump[:, tt, np.newaxis]
        hCb = hCb + dtData * (-hCb + JRb) / tau

        # fixed point driven
        JRc = Jc.dot(Rc[:, tt, np.newaxis])
        JRc += ampInterReg * w_B2C * Rb[:, tt, np.newaxis]
        JRc += ampInterReg * w_A2C * Ra[:, tt, np.newaxis]
        JRc += ampInC * w_Fix2C * hFP[:, tt, np.newaxis]
        hCc = hCc + dtData * (-hCc + JRc) / tau

    # package up outputs
    Rseq = hBump.copy()
    Rfp = hFP.copy()
    # normalize
    Ra = Ra/np.max(Ra)
    Rb = Rb/np.max(Rb)
    Rc = Rc/np.max(Rc)
    Rseq = Rseq/np.max(Rseq)
    Rfp = Rfp/np.max(Rfp)

    out_params = {}
    out_params['Na'] = Na
    out_params['Nb'] = Nb
    out_params['Nc'] = Nc
    out_params['ga'] = ga
    out_params['gb'] = gb
    out_params['gc'] = gc
    out_params['tau'] = tau
    out_params['fracInterReg'] = fracInterReg
    out_params['ampInterReg'] = ampInterReg
    out_params['fracExternal'] = fracExternal
    out_params['ampInB'] = ampInB
    out_params['ampInC'] = ampInC
    out_params['dtData'] = dtData
    out_params['T'] = T
    out_params['leadTime'] = leadTime
    out_params['bumpStd'] = bumpStd

    out = {}
    out['Ra'] = Ra
    out['Rb'] = Rb
    out['Rc'] = Rc
    out['Rseq'] = Rseq
    out['Rfp'] = Rfp
    out['tData'] = tData
    out['Ja'] = Ja
    out['Jb'] = Jb
    out['Jc'] = Jc
    out['w_A2B'] = w_A2B
    out['w_A2C'] = w_A2C
    out['w_B2A'] = w_B2A
    out['w_B2C'] = w_B2C
    out['w_C2A'] = w_C2A
    out['w_C2B'] = w_C2B
    out['w_Fix2C'] = w_Fix2C
    out['w_Seq2B'] = w_Seq2B
    out['params'] = out_params

    if plotSim is True:
        fig = plt.figure(figsize=[8, 8])
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        plt.rcParams.update({'font.size': 6})

        ax = fig.add_subplot(4, 3, 1)
        ax.pcolormesh(tData, range(Na), Ra)
        ax.set_title('RNN A - g={}'.format(ga))

        ax = fig.add_subplot(4, 3, 2)
        ax.pcolormesh(range(Na), range(Na), Ja)
        ax.set_title('DI matrix A')

        ax = fig.add_subplot(4, 3, 3)
        for _ in range(3):
            idx = random.randint(0, Na-1)
            ax.plot(tData, Ra[idx, :])
        ax.set_ylim(-1, 1)
        ax.set_title('units from RNN A')

        ax = fig.add_subplot(4, 3, 4)
        ax.pcolormesh(tData, range(Nb), Rb)
        ax.set_title('RNN B - g={}'.format(gb))

        ax = fig.add_subplot(4, 3, 5)
        ax.pcolormesh(range(Nb), range(Nb), Jb)
        ax.set_title('DI matrix B')

        ax = fig.add_subplot(4, 3, 6)
        for _ in range(3):
            idx = random.randint(0, Nb-1)
            ax.plot(tData, Rb[idx, :])
        ax.set_ylim(-1, 1)
        ax.set_title('units from RNN B')

        ax = fig.add_subplot(4, 3, 7)
        ax.pcolormesh(tData, range(Nc), Rc)
        ax.set_title('RNN C - g={}'.format(gc))

        ax = fig.add_subplot(4, 3, 8)
        ax.pcolormesh(range(Nc), range(Nc), Jc)
        ax.set_title('DI matrix C')

        ax = fig.add_subplot(4, 3, 9)
        for _ in range(3):
            idx = random.randint(0, Nc-1)
            ax.plot(tData, Rc[idx, :])
        ax.set_ylim(-1, 1)
        ax.set_title('units from RNN C')

        ax = fig.add_subplot(4, 3, 10)
        ax.pcolormesh(tData, range(Nc), Rfp)
        ax.set_title('Fixed Point Driver')

        ax = fig.add_subplot(4, 3, 11)
        ax.pcolormesh(tData, range(Nc), Rseq)
        ax.set_title('Sequence Driver')
        plt.pause(0.05)
        fig.show()
    return out

def scaleJ(model):
    scaler = model['scaler']
    train_max = model['train_max']
    scale = scaler.scale_

    J = model['J']
    J = J / scale[np.newaxis, :]
    J = J * scale[:, np.newaxis]

    return J






def computeSumCurrentOut(model, activity):
    # N observations, K region2, O outputs
    idx_reg1 = model['regions']['region1']
    idx_reg2 = model['regions']['region2']

    scaler = model['scaler']
    train_max = model['train_max']

    J_target = model['J'][idx_reg1,][:, idx_reg2] # O x K

    activity_scaled = scaler.inverse_transform(activity.T * train_max)

    activity_reg2 = activity_scaled[:, idx_reg2] #N x K
    activity_reg2_avg = np.average(activity_reg2, axis=0) #1xK


    contribs = J_target * activity_reg2_avg #O x K
    return np.sum(np.abs(contribs), axis=0)


def computeCURBD(sim):
    """
    function [CURBD,CURBDLabels] = computeCURBD(varargin)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % Performs Current-Based Decomposition (CURBD) of multi-region data. Ref:
    %
    % Perich MG et al. Inferring brain-wide interactions using data-constrained
    % recurrent neural network models. bioRxiv. DOI:
    %
    % Two input options:
    %   1) out = computeCURBD(model, params)
    %       Pass in the output struct of trainMultiRegionRNN and it will do the
    %       current decomposition. Note that regions has to be defined.
    %
    %   2) out = computeCURBD(RNN, J, regions, params)
    %       Only needs the RNN activity, region info, and J matrix
    %
    %   Only parameter right now is current_type, to isolate excitatory or
    %   inhibitory currents.
    %
    % OUTPUTS:
    %   CURBD: M x M cell array containing the decomposition for M regions.
    %       Target regions are in rows and source regions are in columns.
    %   CURBDLabels: M x M cell array with string labels for each current
    %
    %
    % Written by Matthew G. Perich. Updated December 2020.
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    current_type = 'all'  # 'excitatory', 'inhibitory', or 'all'
    RNN = sim['RNN']
    J = sim['J'].copy()
    regions = sim['regions']

    if regions is None:
        raise ValueError("regions not specified")

    if current_type == 'excitatory':  # take only positive J weights
        J[J < 0] = 0
    elif current_type == 'inhibitory':  # take only negative J weights
        J[J > 0] = 0
    elif current_type == 'all':
        pass
    else:
        raise ValueError("Unknown current type: {}".format(current_type))

    nRegions = regions.shape[0]

    # loop along all bidirectional pairs of regions
    CURBD = np.empty((nRegions, nRegions), dtype=np.object)
    CURBDLabels = np.empty((nRegions, nRegions), dtype=np.object)

    for idx_trg in range(nRegions):
        in_trg = regions[idx_trg, 1]
        lab_trg = regions[idx_trg, 0]
        for idx_src in range(nRegions):
            in_src = regions[idx_src, 1]
            lab_src = regions[idx_src, 0]
            sub_J = J[in_trg, :][:, in_src]
            CURBD[idx_trg, idx_src] = sub_J.dot(RNN[in_src, :])
            CURBDLabels[idx_trg, idx_src] = "{} to {}".format(lab_src,
                                                              lab_trg)
    return (CURBD, CURBDLabels)


def region_currs(model, rates):
    tau = model['params']['tauRNN']
    idx_reg1 = model['regions']['region1']
    idx_reg2 = model['regions']['region2']
    J = scaleJ(model)
    RNN_reg1 = rates[idx_reg1,:]
    RNN_reg2 = rates[idx_reg2,:]

    J_dstream = J[idx_reg1,:]
    dstream = J_dstream[:, idx_reg1]
    upstream = J_dstream[:, idx_reg2]
    
    dstream_inp = dstream@RNN_reg1 / tau
    upstream_inp = upstream@RNN_reg2 / tau

    dstream_decay = ((tau-1)/tau)*RNN_reg1

    return dstream_inp.T, upstream_inp.T, dstream_decay.T

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



