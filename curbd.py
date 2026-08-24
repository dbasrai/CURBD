""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Performs Current-Based Decomposition (CURBD) of multi-region data. Ref:
%
% Perich MG et al. Inferring brain-wide interactions using data-constrained
% recurrent neural network models. bioRxiv. DOI: https://doi.org/10.1101/2020.12.18.423348
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

import math
import os
import pickle
import random
import re

import numpy as np
import numpy.random as npr
import numpy.linalg

import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
try:
    from src.utils.evaluate import *
except ImportError:
    pass


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


EI_POP_ORDER = ('CFA_E', 'CFA_I', 'RFA_E', 'RFA_I')


def _causal_gaussian_smooth_trial(trial, sigma):
    """Causal Gaussian smooth of a (T, N) trial along time."""
    trial = np.asarray(trial, dtype=float)
    if sigma is None or sigma <= 0:
        return trial
    radius = max(1, int(np.ceil(3.0 * float(sigma))))
    t = np.arange(0, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (t / float(sigma)) ** 2)
    kernel /= kernel.sum()
    out = np.empty_like(trial)
    for n in range(trial.shape[1]):
        conv = np.convolve(trial[:, n], kernel, mode='full')
        out[:, n] = conv[:trial.shape[0]]
    return out


def _zscore_activity(activity_tn):
    """Z-score (T, N) activity. Returns (z_tn, scaler) with inverse_transform."""
    try:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        z_tn = scaler.fit_transform(activity_tn)
        return z_tn, scaler
    except ImportError:
        mean = activity_tn.mean(axis=0)
        std = activity_tn.std(axis=0)
        std = np.where(std == 0, 1.0, std)

        class _MeanStdScaler(object):
            def __init__(self, mean_, scale_):
                self.mean_ = mean_
                self.scale_ = scale_

            def inverse_transform(self, x):
                return x * self.scale_ + self.mean_

        z_tn = (activity_tn - mean) / std
        return z_tn, _MeanStdScaler(mean, std)


def _load_yaml_dict(path):
    if path is None or not os.path.isfile(path):
        return None
    try:
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _binsize_ms_from_name(path):
    match = re.search(r'binsize(\d+)', os.path.basename(str(path)))
    if match:
        return int(match.group(1))
    return None


def _pre_ms_from_name(path):
    match = re.search(r'pre(\d+)ms', os.path.basename(str(path)))
    if match:
        return int(match.group(1))
    return None


def _opto_fields_from_cfgs(yaml_cfg, snapshot_cfg, path=None):
    """opto_target_population is CFA_I (co9) or RFA_I (co10/co12), not always region2."""
    merged = {}
    for src in (yaml_cfg or {}, snapshot_cfg or {}):
        for key in ('opto_target_population', 'target_population',
                    'opto_corresponding_e_population', 'stimulated_region'):
            if src.get(key) and key not in merged:
                merged[key] = src[key]
    target = (merged.get('opto_target_population')
              or merged.get('target_population'))
    if target is None and merged.get('stimulated_region'):
        target = '{}_I'.format(merged['stimulated_region'])
    e_pop = merged.get('opto_corresponding_e_population')
    if e_pop is None and target and str(target).endswith('_I'):
        e_pop = target[:-2] + '_E'
    region = merged.get('stimulated_region')
    if region is None and target:
        region = str(target).split('_')[0]
    return {
        'opto_target_population': target,
        'opto_corresponding_e_population': e_pop,
        'stimulated_region': region,
        'stim_onset_s': (_pre_ms_from_name(path) or 20) / 1000.0,
    }


def _resolve_ei_dataset_paths(path):
    path = os.path.abspath(os.path.expanduser(str(path)))
    if path.endswith('.yaml'):
        stem = path[:-5]
        if stem.endswith('_preprocessing_snapshot'):
            stem = stem[: -len('_preprocessing_snapshot')]
        pkl_path = stem + '.pkl'
        yaml_path = stem + '.yaml'
    elif path.endswith('.pkl'):
        stem = path[:-4]
        pkl_path = path
        yaml_path = stem + '.yaml'
    else:
        stem = path
        pkl_path = stem + '.pkl'
        yaml_path = stem + '.yaml'
    snapshot_path = stem + '_preprocessing_snapshot.yaml'
    return pkl_path, yaml_path, snapshot_path


def ei_sign_from_populations(populations, number_units):
    ei_sign = np.ones(number_units, dtype=float)
    for name, idx in populations.items():
        idx = np.asarray(idx)
        if name.endswith('_I'):
            ei_sign[idx] = -1.0
        else:
            ei_sign[idx] = 1.0
    return ei_sign


def _dale_masks(number_units, num_reg1, ei_sign):
    ei_sign = np.asarray(ei_sign).reshape(-1)
    e_col = (ei_sign > 0)[np.newaxis, :]
    i_col = (ei_sign < 0)[np.newaxis, :]
    intra = np.zeros((number_units, number_units), dtype=bool)
    intra[:num_reg1, :num_reg1] = True
    intra[num_reg1:, num_reg1:] = True
    inter = np.zeros((number_units, number_units), dtype=bool)
    inter[:num_reg1, num_reg1:] = True
    inter[num_reg1:, :num_reg1] = True
    return {
        'intra_e': intra & e_col,
        'intra_i': intra & i_col,
        'inter_e': inter & e_col,
        'inter_i': inter & i_col,
    }


def _project_inter_inplace(J, masks):
    """Hard inter-region constraints: E>=0, I→across identically 0."""
    J[masks['inter_e']] = np.maximum(J[masks['inter_e']], 0.0)
    J[masks['inter_i']] = 0.0
    return J


def _project_intra_inplace(J, masks, alpha=1.0):
    """Leak intra Dale violations toward 0. alpha=0 leaves them; 1 hard-clips."""
    alpha = float(alpha)
    if alpha <= 0.0:
        return J
    if alpha >= 1.0:
        J[masks['intra_e']] = np.maximum(J[masks['intra_e']], 0.0)
        J[masks['intra_i']] = np.minimum(J[masks['intra_i']], 0.0)
        return J
    e = J[masks['intra_e']]
    i = J[masks['intra_i']]
    e = np.where(e < 0.0, e * (1.0 - alpha), e)
    i = np.where(i > 0.0, i * (1.0 - alpha), i)
    J[masks['intra_e']] = e
    J[masks['intra_i']] = i
    return J


def _project_dale_inplace(J, masks, alpha=1.0):
    _project_inter_inplace(J, masks)
    _project_intra_inplace(J, masks, alpha=alpha)
    return J


def _illegal_intra_mass(J, masks):
    """L2 mass of intra weights that violate Dale (E<0 or I>0)."""
    e = np.asarray(J)[masks['intra_e']]
    i = np.asarray(J)[masks['intra_i']]
    return float(np.sum(np.minimum(e, 0.0) ** 2) + np.sum(np.maximum(i, 0.0) ** 2))


def _normalize_intra_dale_mode(intra_dale):
    if intra_dale in (False, 0, 'off', 'none'):
        return 'off'
    if intra_dale in (True, 1, 'hard', 'on'):
        return 'hard'
    if intra_dale in ('anneal', 'ramp'):
        return 'anneal'
    raise ValueError('Unknown intra_dale: {}'.format(intra_dale))


def _intra_dale_alpha(nRun, nRunTrain, intra_dale, ramp=(0.3, 0.8)):
    """Dale leak strength for this run. Free runs use 1 unless intra_dale is off."""
    mode = _normalize_intra_dale_mode(intra_dale)
    if nRun >= nRunTrain:
        return 0.0 if mode == 'off' else 1.0
    if mode == 'off':
        return 0.0
    if mode == 'hard':
        return 1.0
    start_frac, end_frac = ramp
    start_frac = float(start_frac)
    end_frac = float(end_frac)
    if end_frac <= start_frac:
        end_frac = min(1.0, start_frac + 1e-6)
    frac = (nRun + 1) / float(max(int(nRunTrain), 1))
    if frac <= start_frac:
        return 0.0
    if frac >= end_frac:
        return 1.0
    return (frac - start_frac) / (end_frac - start_frac)


def _radius_cap_this_run(nRun, nRunTrain, max_radius, max_radius_after_frac):
    """Return max_radius if the cap is active this run, else None."""
    if max_radius is None:
        return None
    if max_radius_after_frac is None:
        return float(max_radius)
    frac = (min(nRun, max(int(nRunTrain) - 1, 0)) + 1) / float(max(int(nRunTrain), 1))
    if nRun >= nRunTrain:
        return float(max_radius)
    if frac < float(max_radius_after_frac):
        return None
    return float(max_radius)


def _maybe_cap_radius(J, masks, max_radius, alpha):
    if max_radius is None:
        return J
    rho = _spectral_radius(J)
    if rho > max_radius and rho > 0:
        J *= float(max_radius) / rho
        _project_inter_inplace(J, masks)
        _project_intra_inplace(J, masks, alpha=alpha)
    return J


def _init_dale_J(number_units, num_reg1, ei_sign, g, g_across, g_loc,
                 init_intra='dale'):
    """Intra init plus sparse-excitatory E-only inter-region.

    init_intra='dale': folded-Gaussian E>=0 / I<=0 (hard Dale from the start).
    init_intra='signed': Bio-style signed Gaussian intra (no fold); Dale is
    applied later by annealing.
    """
    num_reg2 = number_units - num_reg1
    ei_sign = np.asarray(ei_sign).reshape(-1)
    e1 = np.where(ei_sign[:num_reg1] > 0)[0]
    i1 = np.where(ei_sign[:num_reg1] < 0)[0]
    e2 = np.where(ei_sign[num_reg1:] > 0)[0] + num_reg1
    i2 = np.where(ei_sign[num_reg1:] < 0)[0] + num_reg1

    J = np.zeros((number_units, number_units), dtype=float)
    if init_intra == 'signed':
        J[:num_reg1, :num_reg1] = g * npr.randn(num_reg1, num_reg1)
        J[num_reg1:, num_reg1:] = g * npr.randn(num_reg2, num_reg2)
        J[:num_reg1, :num_reg1] = J[:num_reg1, :num_reg1] + g_loc[0]
        J[num_reg1:, num_reg1:] = J[num_reg1:, num_reg1:] + g_loc[1]
    elif init_intra == 'dale':
        if len(e1):
            J[:num_reg1, e1] = g * np.abs(npr.randn(num_reg1, len(e1)))
        if len(i1):
            J[:num_reg1, i1] = -g * np.abs(npr.randn(num_reg1, len(i1)))
        J[:num_reg1, :num_reg1] = J[:num_reg1, :num_reg1] + g_loc[0]
        if len(e1):
            J[:num_reg1, e1] = np.maximum(J[:num_reg1, e1], 0.0)
        if len(i1):
            J[:num_reg1, i1] = np.minimum(J[:num_reg1, i1], 0.0)

        if len(e2):
            J[num_reg1:, e2] = g * np.abs(npr.randn(num_reg2, len(e2)))
        if len(i2):
            J[num_reg1:, i2] = -g * np.abs(npr.randn(num_reg2, len(i2)))
        J[num_reg1:, num_reg1:] = J[num_reg1:, num_reg1:] + g_loc[1]
        if len(e2):
            J[num_reg1:, e2] = np.maximum(J[num_reg1:, e2], 0.0)
        if len(i2):
            J[num_reg1:, i2] = np.minimum(J[num_reg1:, i2], 0.0)
    else:
        raise ValueError('Unknown init_intra: {}'.format(init_intra))

    # Half-normal via |N(0,1)| so numpy.random.seed controls the whole J0.
    if len(e2):
        J[:num_reg1, e2] = (g_across / g) * np.abs(npr.randn(num_reg1, len(e2)))
    if len(e1):
        J[num_reg1:, e1] = (g_across / g) * np.abs(npr.randn(num_reg2, len(e1)))

    J = J / math.sqrt(number_units)
    return J


def load_ei_dataset(path, dtFactor=5, smooth_sigma=1.5, zscore=True,
                    max_trials=None, reset_every=None):
    """Load CFA/RFA x E/I trial pickles into CURBD training arrays.

    Parameters
    ----------
    path : str
        Dataset stem, .pkl, or .yaml under datasets/.
    dtFactor : int
        RNN interpolation factor; used to place trial-boundary resetPoints.
    smooth_sigma : float
        Causal Gaussian sigma in bins, applied within each trial. 0 disables.
    zscore : bool
        If True, StandardScaler along time (same as curbdSweepHPs.py).

    Returns
    -------
    dict with z_activity (N x T), scaler, regions, populations, ei_sign,
    resetPoints, dtData, and metadata.
    """
    pkl_path, yaml_path, snapshot_path = _resolve_ei_dataset_paths(path)
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError('Dataset pickle not found: {}'.format(pkl_path))

    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)
    missing = [k for k in EI_POP_ORDER if k not in raw]
    if missing:
        raise KeyError('Dataset missing populations {}: {}'.format(missing, pkl_path))

    n_trials_all = len(raw['CFA_E'])
    for name in EI_POP_ORDER:
        if len(raw[name]) != n_trials_all:
            raise ValueError('Trial count mismatch for {}'.format(name))
    n_trials = n_trials_all
    if max_trials is not None:
        n_trials = min(n_trials, int(max_trials))
    trial_length = np.asarray(raw['CFA_E'][0]).shape[0]
    pop_counts = {}
    for name in EI_POP_ORDER:
        pop_counts[name] = np.asarray(raw[name][0]).shape[1]

    yaml_cfg = _load_yaml_dict(yaml_path) or {}
    snapshot_cfg = _load_yaml_dict(snapshot_path) or {}
    yaml_neurons = yaml_cfg.get('neurons') or snapshot_cfg.get('neurons') or {}
    opto_meta = _opto_fields_from_cfgs(yaml_cfg, snapshot_cfg, pkl_path)

    offset = 0
    populations = {}
    for name in EI_POP_ORDER:
        n = pop_counts[name]
        populations[name] = np.arange(offset, offset + n)
        if name in yaml_neurons:
            lo, hi = yaml_neurons[name]
            if (lo, hi) != (int(offset), int(offset + n)):
                print('Warning: YAML neurons[{}]={} but pickle concat is [{}, {})'.format(
                    name, yaml_neurons[name], offset, offset + n))
        offset += n
    number_units = offset

    trials = []
    for t in range(n_trials):
        blocks = [_causal_gaussian_smooth_trial(np.asarray(raw[name][t]), smooth_sigma)
                  for name in EI_POP_ORDER]
        trial = np.hstack(blocks)
        if trial.shape != (trial_length, number_units):
            raise ValueError('Unexpected trial shape {} at trial {}'.format(trial.shape, t))
        trials.append(trial)
    activity_tn = np.concatenate(trials, axis=0)

    binsize_ms = (yaml_cfg.get('binsize') or snapshot_cfg.get('binsize')
                  or _binsize_ms_from_name(pkl_path) or 20)
    dtData = float(binsize_ms) / 1000.0

    scaler = None
    if zscore:
        activity_tn, scaler = _zscore_activity(activity_tn)
    z_activity = activity_tn.T

    n_cfa = len(populations['CFA_E']) + len(populations['CFA_I'])
    n_rfa = len(populations['RFA_E']) + len(populations['RFA_I'])
    regions = {
        'region1': np.arange(0, n_cfa),
        'region2': np.arange(n_cfa, n_cfa + n_rfa),
    }
    ei_sign = ei_sign_from_populations(populations, number_units)
    resetPoints = make_reset_points(
        n_trials, int(trial_length), int(dtFactor), reset_every=reset_every)

    return {
        'z_activity': z_activity,
        'scaler': scaler,
        'regions': regions,
        'populations': populations,
        'ei_sign': ei_sign,
        'resetPoints': resetPoints,
        'dtData': dtData,
        'binsize_ms': int(binsize_ms),
        'n_trials': n_trials,
        'trial_length': int(trial_length),
        'pkl_path': pkl_path,
        'name': os.path.splitext(os.path.basename(pkl_path))[0],
        'opto_target_population': opto_meta['opto_target_population'],
        'opto_corresponding_e_population': opto_meta['opto_corresponding_e_population'],
        'stimulated_region': opto_meta['stimulated_region'],
        'stim_onset_s': opto_meta['stim_onset_s'],
    }


def make_reset_points(n_trials, trial_length, dtFactor, reset_every=None):
    """RNN-time indices to teacher-force H from data.

    Trial starts are always included. If reset_every is set (in RNN steps),
    extra resets are inserted inside each trial, matching curbdSweepHPs
    num_reset=100 on the old 5–10 ms pipeline.
    """
    trial_rnn = int(trial_length) * int(dtFactor)
    starts = np.arange(int(n_trials), dtype=np.int32) * trial_rnn
    if reset_every is None or int(reset_every) <= 0:
        return starts
    chunks = [np.arange(int(s), int(s) + trial_rnn, int(reset_every), dtype=np.int32)
              for s in starts]
    return np.unique(np.concatenate(chunks)).astype(np.int32)


def _hidden_from_rates(rates_col, reset_state, nonLinearity_inv):
    """Map a data rate vector to RNN hidden state H at a reset."""
    col = np.asarray(rates_col, dtype=float)
    if col.ndim == 1:
        col = col[:, None]
    if reset_state == 'current':
        return nonLinearity_inv(np.clip(col, -0.999, 0.999))
    return col


def _prepare_adata(activity, adata_scale='z3'):
    """Put z-scored (or raw) activity into a tanh-friendly range.

    'max' matches original CURBD but is dominated by outliers after z-scoring.
    'z3' divides by 3 (typical z-score range) then clips.
    'p99' divides by the 99th percentile of |activity|.
    """
    Adata = np.asarray(activity, dtype=float).copy()
    if adata_scale == 'max':
        train_max = float(np.max(np.abs(Adata))) or 1.0
        Adata = Adata / train_max
    elif adata_scale == 'p99':
        train_max = float(np.percentile(np.abs(Adata), 99)) or 1.0
        Adata = Adata / train_max
    elif adata_scale in ('z3', 'clip3'):
        train_max = 3.0
        Adata = Adata / train_max
    elif adata_scale == 'clip':
        train_max = 1.0
    else:
        raise ValueError('Unknown adata_scale: {}'.format(adata_scale))
    Adata = np.clip(Adata, -0.999, 0.999)
    return Adata, train_max


def _spectral_radius(J):
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(J)))))


def _softplus(x):
    """Stable softplus: log(1 + exp(x))."""
    return np.logaddexp(0.0, np.asarray(x, dtype=float))


def _sigmoid(x):
    x = np.clip(np.asarray(x, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _inv_softplus(y, eps=1e-12):
    """Inverse of softplus, defined for y > 0."""
    y = np.maximum(np.asarray(y, dtype=float), eps)
    out = np.empty_like(y)
    big = y > 20.0
    out[big] = y[big]
    small = ~big
    out[small] = np.log(np.expm1(y[small]))
    return out


def _i_across_col_masks(gamma, num_reg1):
    """Boolean source-column masks for I cells (gamma < 0) in each region."""
    cols = np.arange(gamma.shape[0])
    rfa_i = (gamma < 0) & (cols >= num_reg1)
    cfa_i = (gamma < 0) & (cols < num_reg1)
    return cfa_i, rfa_i


def _intra_column_mean(J, num_reg1):
    """Per-source mean of intra weights. J[i, j] = source j → target i."""
    n = J.shape[0]
    score = np.zeros(n, dtype=float)
    score[:num_reg1] = np.asarray(J)[:num_reg1, :num_reg1].mean(axis=0)
    score[num_reg1:] = np.asarray(J)[num_reg1:, num_reg1:].mean(axis=0)
    return score


def _n_e_from_targets(n_region, e_frac=None, n_E=None):
    if n_E is not None:
        k = int(n_E)
    elif e_frac is not None:
        k = int(round(float(e_frac) * n_region))
    else:
        k = int(round(0.5 * n_region))
    return int(np.clip(k, 0, n_region))


def gamma_ranked_from_J(J, num_reg1, e_frac=None, n_E=None):
    """E = highest intra-mean columns in each region, with a count target.

    Unlike i0_sign, this can mark a column E even if its I0 mean is negative,
    so CFA/RFA E counts are chosen rather than inferred from sign(mean).
    e_frac: scalar or (cfa, rfa). n_E: (n_cfa_e, n_rfa_e). n_E wins if both set.
    """
    n = J.shape[0]
    score = _intra_column_mean(J, num_reg1)
    if n_E is None:
        if e_frac is None:
            e_frac = 0.5
        if np.isscalar(e_frac):
            f1 = f2 = float(e_frac)
        else:
            f1, f2 = float(e_frac[0]), float(e_frac[1])
        k1 = _n_e_from_targets(num_reg1, e_frac=f1)
        k2 = _n_e_from_targets(n - num_reg1, e_frac=f2)
    else:
        k1 = _n_e_from_targets(num_reg1, n_E=n_E[0])
        k2 = _n_e_from_targets(n - num_reg1, n_E=n_E[1])
    gamma = -np.ones(n, dtype=float)
    gamma[np.argsort(-score[:num_reg1])[:k1]] = 1.0
    gamma[num_reg1 + np.argsort(-score[num_reg1:])[:k2]] = 1.0
    return gamma


def gamma_from_init(J, num_reg1, ei_sign=None, mode='pickle',
                    e_frac=None, n_E=None):
    """Per-source Dale scalar. J[i, j] is source j → target i, so gamma is a column scale."""
    n = J.shape[0]
    if mode == 'i0_sign':
        gamma = np.sign(_intra_column_mean(J, num_reg1))
        gamma[gamma == 0] = 1.0
        return gamma.astype(float)
    if mode in ('i0_rank', 'ranked', 'balanced'):
        return gamma_ranked_from_J(J, num_reg1, e_frac=e_frac, n_E=n_E)
    if ei_sign is None:
        raise ValueError('gamma_from_init mode pickle needs ei_sign')
    gamma = np.where(np.asarray(ei_sign).reshape(-1) > 0, 1.0, -1.0)
    gamma[gamma == 0] = 1.0
    return gamma.astype(float)


def j_from_dale_scalar(W, gamma, num_reg1, zero_i_across=True):
    """J[:, j] = gamma[j] * softplus(W[:, j]). I sources (gamma<0) do not project across."""
    J = np.asarray(gamma, dtype=float).reshape(1, -1) * _softplus(W)
    if zero_i_across:
        cfa_i, rfa_i = _i_across_col_masks(gamma, num_reg1)
        J[num_reg1:, cfa_i] = 0.0
        J[:num_reg1, rfa_i] = 0.0
    return J


def _init_W_from_J_gamma(J, gamma, num_reg1, zero_i_across=True, eps=1e-8):
    """Project J onto gamma * R_+ and invert softplus for W."""
    gamma = np.asarray(gamma, dtype=float).reshape(-1)
    tiny = np.abs(gamma) < 1e-8
    gamma = gamma.copy()
    gamma[tiny] = np.where(gamma[tiny] >= 0, 1e-3, -1e-3)
    target = np.maximum(J / gamma.reshape(1, -1), eps)
    if zero_i_across:
        cfa_i, rfa_i = _i_across_col_masks(gamma, num_reg1)
        target[num_reg1:, cfa_i] = eps
        target[:num_reg1, rfa_i] = eps
    return _inv_softplus(target)


def _dale_intra_cone_l2(J_unc, num_reg1):
    """Per-source L2 mass of intra weights on the E cone vs the I cone.

    J is source-column oriented (J[i, j] = j → i). Identity is an intra
    property: E keeps positive intra, I keeps negative intra.
    """
    J_unc = np.asarray(J_unc, dtype=float)
    n = J_unc.shape[0]
    intra = np.zeros((n, n), dtype=float)
    intra[:num_reg1, :num_reg1] = J_unc[:num_reg1, :num_reg1]
    intra[num_reg1:, num_reg1:] = J_unc[num_reg1:, num_reg1:]
    e_l2 = np.sum(np.maximum(intra, 0.0) ** 2, axis=0)
    i_l2 = np.sum(np.maximum(-intra, 0.0) ** 2, axis=0)
    return e_l2, i_l2


def _enforce_min_e_gamma(gamma_old, gamma_new, num_reg1, min_e_count, pref_i=None):
    """Keep at least min_e_count E sources per region after a sign update."""
    if min_e_count is None:
        return gamma_new
    gamma_new = np.asarray(gamma_new, dtype=float).copy()
    old_sign = np.sign(gamma_old)
    old_sign[old_sign == 0] = 1.0
    new_sign = np.sign(gamma_new)
    new_sign[new_sign == 0] = np.sign(old_sign[new_sign == 0])
    n = gamma_new.size
    if pref_i is None:
        pref_i = -np.abs(gamma_new)
    splits = [(0, num_reg1, int(min_e_count[0])),
              (num_reg1, n, int(min_e_count[1]))]
    for lo, hi, kmin in splits:
        if int(np.sum(new_sign[lo:hi] > 0)) >= kmin:
            continue
        became_i = lo + np.where(
            (old_sign[lo:hi] > 0) & (new_sign[lo:hi] < 0))[0]
        if len(became_i) == 0:
            continue
        order = became_i[np.argsort(pref_i[became_i])]
        need = kmin - int(np.sum(new_sign[lo:hi] > 0))
        keep = order[:need]
        new_sign[keep] = 1.0
        gamma_new[keep] = np.maximum(np.abs(gamma_new[keep]), 1e-3)
    return gamma_new


def dale_scalar_reassign_from_unc(W, gamma, J_unc, num_reg1, zero_i_across=True,
                                  margin=0.05, eps=1e-8, dJ_acc=None,
                                  min_update_l2=1e-8, min_e_count=None):
    """Flip gamma when FORCE prefers the other intra cone.

    Score the accumulated FORCE update dJ_acc when given. Scoring J_unc itself
    almost never flips: the current legal column dominates J + dJ.
    Rebuilds W by projecting J_unc onto the chosen cone. Flipped units get
    |gamma|=1 so scale lives in W instead of riding the previous |gamma| to ±10.
    min_e_count: (n_cfa_e, n_rfa_e) floor. E→I flips that would go below
    the floor are blocked.
    """
    gamma = np.asarray(gamma, dtype=float).reshape(-1)
    J_unc = np.asarray(J_unc, dtype=float)
    score_J = dJ_acc if dJ_acc is not None else J_unc
    e_l2, i_l2 = _dale_intra_cone_l2(score_J, num_reg1)
    old_sign = np.sign(gamma)
    old_sign[old_sign == 0] = 1.0
    new_sign = old_sign.copy()
    strong = (e_l2 + i_l2) > float(min_update_l2)
    new_sign[strong & (i_l2 > e_l2 * (1.0 + margin)) & (old_sign > 0)] = -1.0
    new_sign[strong & (e_l2 > i_l2 * (1.0 + margin)) & (old_sign < 0)] = 1.0
    gamma_new = new_sign * np.abs(gamma)
    flipped = new_sign != old_sign
    gamma_new[flipped] = new_sign[flipped]
    gamma_new = _enforce_min_e_gamma(
        gamma, gamma_new, num_reg1, min_e_count, pref_i=(i_l2 - e_l2))
    W = _init_W_from_J_gamma(J_unc, gamma_new, num_reg1, zero_i_across, eps)
    J = j_from_dale_scalar(W, gamma_new, num_reg1, zero_i_across)
    n_flip = int(np.sum(np.sign(gamma_new) != old_sign))
    return W, gamma_new, J, n_flip


def dale_scalar_force_step(W, gamma, dJ, num_reg1, zero_i_across=True,
                           eps=1e-8, gamma_clip=10.0, gamma_gain=1.0,
                           min_e_count=None, gamma_l2=0.0):
    """FORCE step on {gamma, W}: least-squares gamma, then project J+dJ onto gamma * R_+.

    Each source column stays single-signed. gamma is unconstrained and can flip E↔I.
    gamma_gain > 1 amplifies the LS gamma step so signs can actually cross 0.
    min_e_count blocks E→I crossings that would drop a region below its E floor.
    gamma_l2: ridge of |gamma| toward 1, scaled by median ||softplus(W)||^2 so it
        mainly rescues collapsed-W columns (the ±10 clip) rather than shrinking
        healthy |gamma|. 0 = vanilla LS.
    """
    gamma = np.asarray(gamma, dtype=float).reshape(-1)
    sp = _softplus(W)
    J_old = gamma.reshape(1, -1) * sp
    denom = np.sum(sp * sp, axis=0) + eps
    gamma_ls = gamma + float(gamma_gain) * np.sum(dJ * sp, axis=0) / denom
    lam = float(gamma_l2) * max(float(np.median(denom)), 1e-4)
    if lam > 0:
        t = np.sign(gamma)
        t[t == 0] = 1.0
        gamma_new = (gamma_ls * denom + lam * t) / (denom + lam)
    else:
        gamma_new = gamma_ls
    gamma_new = np.clip(gamma_new, -gamma_clip, gamma_clip)
    J_unc = J_old + dJ
    tiny = np.abs(gamma_new) < 1e-4
    if np.any(tiny):
        e_l2, i_l2 = _dale_intra_cone_l2(J_unc, num_reg1)
        s = np.where(e_l2[tiny] >= i_l2[tiny], 1.0, -1.0)
        gamma_new[tiny] = s * 1e-3
    e_l2, i_l2 = _dale_intra_cone_l2(J_unc, num_reg1)
    gamma_new = _enforce_min_e_gamma(
        gamma, gamma_new, num_reg1, min_e_count, pref_i=(i_l2 - e_l2))
    target = np.maximum(J_unc / gamma_new.reshape(1, -1), eps)
    if zero_i_across:
        cfa_i, rfa_i = _i_across_col_masks(gamma_new, num_reg1)
        target[num_reg1:, cfa_i] = eps
        target[:num_reg1, rfa_i] = eps
    W = _inv_softplus(target)
    J = j_from_dale_scalar(W, gamma_new, num_reg1, zero_i_across)
    return W, gamma_new, J


def constraint_block_metrics(J, ei_sign, num_reg1, atol=1e-12):
    """I→across L1, intra Dale violation fraction, and negative-cross fraction."""
    J = np.asarray(J)
    masks = _dale_masks(J.shape[0], num_reg1, ei_sign)
    intra_e = J[masks['intra_e']]
    intra_i = J[masks['intra_i']]
    inter_i = J[masks['inter_i']]
    n_intra = int(intra_e.size + intra_i.size)
    n_viol = int(np.sum(intra_e < -atol) + np.sum(intra_i > atol))
    cross = np.concatenate([J[:num_reg1, num_reg1:].ravel(),
                            J[num_reg1:, :num_reg1].ravel()])
    mixed_col = (np.any(J > atol, axis=0) & np.any(J < -atol, axis=0))
    return {
        'i_across_l1': float(np.sum(np.abs(inter_i))) if inter_i.size else 0.0,
        'intra_sign_viol_frac': float(n_viol / max(n_intra, 1)),
        'cross_neg_frac': float(np.mean(cross < 0.0)) if cross.size else 0.0,
        'mixed_col_frac': float(np.mean(mixed_col)),
    }


def trainBioConstrainedRNN(activity, dtData=1, dtFactor=1, g=1.5, tauRNN=0.01,
                        tauWN=0.1, ampInWN=0.01, nRunTrain=2000,
                        nRunFree=10, P0=1.0,
                        nonLinearity=np.tanh,
                        nonLinearity_inv=np.arctanh,
                        resetPoints=None,
                        plotStatus=True, verbose=True,
                        regions=None, g_across=None,
                        sparse_percent=80,
                        g_loc=(0, 0),
                        ei_sign=None,
                        populations=None,
                        zero_i_across=False,
                        intra_reparam=False,
                        dale_scalar=False,
                        J_init=None,
                        gamma_init='pickle',
                        gamma_gain=1.0,
                        epoch_flip=False,
                        flip_every=None,
                        flip_margin=0.05,
                        e_frac=None,
                        n_E=None,
                        min_e_frac=None,
                        gamma_l2=0.0):
    """Bio FORCE loop with sequential constraints.

    zero_i_across: I columns do not project to the other region (identically 0);
        remaining E cross stays Bio-style (truncnorm, clip negatives at reset,
        sparsify weakest E source columns). Intra J stays signed.

    intra_reparam: requires zero_i_across. Intra Dale via
        J_E = softplus(W), J_I = -softplus(W); FORCE updates W by the chain
        rule. Cross E is still Bio clip, not reparameterized. Labels are frozen.

    dale_scalar: J[:, j] = gamma[j] * softplus(W[:, j]). Each source is purely
        E or I (never mixed signs). gamma is an unconstrained learned scalar
        and may flip identity. I-across follows gamma < 0 if zero_i_across.

    gamma_gain: scale the FORCE least-squares step on gamma (1 = vanilla LS).
    gamma_l2: ridge |gamma| toward 1 during the FORCE LS step (0 = off).
        Identity flips always rebuild with |gamma|=1; scale goes into W.
    epoch_flip: after each training run, reassign signs from unconstrained
        intra cone mass (J_anchor + accumulated dJ).
    flip_every: also reassign every this many data bins during a run (e.g. 41
        = once per trial). None = only epoch_flip / LS crossing.
    e_frac / n_E: for gamma_init i0_rank, per-region E fraction or counts.
    min_e_frac: during epoch_flip, do not let a region fall below this E fraction.
    """
    if intra_reparam and dale_scalar:
        raise ValueError('intra_reparam and dale_scalar are mutually exclusive')
    if intra_reparam and not zero_i_across:
        raise ValueError(
            'intra_reparam is sequential on zero I→across; set zero_i_across=True')
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
    if isinstance(g_loc, (int, float)):
        g_loc = (g_loc, g_loc)

    number_units = activity.shape[0]
    number_learn = activity.shape[0]

    if populations is None:
        populations = {}
    if ei_sign is None:
        if zero_i_across or intra_reparam or dale_scalar:
            if not populations:
                raise ValueError(
                    'trainBioConstrainedRNN needs ei_sign or populations '
                    'when zero_i_across, intra_reparam, or dale_scalar is set')
            ei_sign = ei_sign_from_populations(populations, number_units)
        else:
            ei_sign = np.ones(number_units, dtype=float)
    ei_sign = np.asarray(ei_sign, dtype=float).reshape(-1)
    if ei_sign.shape[0] != number_units:
        raise ValueError('ei_sign length {} != number_units {}'.format(
            ei_sign.shape[0], number_units))

    masks = _dale_masks(number_units, num_reg1, ei_sign)
    e1_rel = np.where(ei_sign[:num_reg1] > 0)[0]
    e2_rel = np.where(ei_sign[num_reg1:] > 0)[0]

    dtRNN = dtData / float(dtFactor)
    nRunTot = nRunTrain + nRunFree
    percentile_list = np.linspace(sparse_percent // 2, sparse_percent, nRunTrain)
    print(percentile_list)

    learnList = npr.permutation(number_units)
    iTarget = learnList[:number_learn]
    iNonTarget = learnList[number_learn:]
    tData = dtData * np.arange(activity.shape[1])
    tRNN = np.arange(0, tData[-1] + dtRNN, dtRNN)

    ampWN = math.sqrt(tauWN / dtRNN)
    iWN = ampWN * npr.randn(number_units, len(tRNN))
    inputWN = np.ones((number_units, len(tRNN)))
    for tt in range(1, len(tRNN)):
        inputWN[:, tt] = iWN[:, tt] + (inputWN[:, tt - 1] - iWN[:, tt]) * np.exp(-(dtRNN / tauWN))
    inputWN = ampInWN * inputWN

    J = g * (npr.randn(number_units, number_units))
    J[:num_reg1, :num_reg1] = J[:num_reg1, :num_reg1] + g_loc[0]
    J[num_reg1:, num_reg1:] = J[num_reg1:, num_reg1:] + g_loc[1]
    J[:num_reg1, num_reg1:] = (g_across / g) * (truncnorm.rvs(
        a=0, b=np.inf, loc=0, scale=1, size=(num_reg1, num_reg2)))
    J[num_reg1:, :num_reg1] = (g_across / g) * (truncnorm.rvs(
        a=0, b=np.inf, loc=0, scale=1, size=(num_reg2, num_reg1)))
    J = J / math.sqrt(number_units)
    if J_init is not None:
        J = np.asarray(J_init, dtype=float).copy()
        if J.shape != (number_units, number_units):
            raise ValueError('J_init shape {} != ({}, {})'.format(
                J.shape, number_units, number_units))

    if zero_i_across and not dale_scalar:
        J[masks['inter_i']] = 0.0

    W = None
    gamma = None
    gamma0 = None
    ei_sign_init = ei_sign.copy()
    n_E_hist = []
    pickle_agree_hist = []
    n_flip_hist = []
    n_E_kw = n_E
    if n_E_kw == 'pickle' and populations:
        n_E_kw = (
            len(np.asarray(populations['CFA_E'])),
            len(np.asarray(populations['RFA_E'])),
        )
    min_e_count = None
    if min_e_frac is not None:
        if np.isscalar(min_e_frac):
            f1 = f2 = float(min_e_frac)
        else:
            f1, f2 = float(min_e_frac[0]), float(min_e_frac[1])
        min_e_count = (
            _n_e_from_targets(num_reg1, e_frac=f1),
            _n_e_from_targets(number_units - num_reg1, e_frac=f2),
        )
    if dale_scalar:
        if np.ndim(gamma_init) > 0:
            gamma = np.asarray(gamma_init, dtype=float).reshape(-1)
            if gamma.shape[0] != number_units:
                raise ValueError('gamma_init length {} != {}'.format(
                    gamma.shape[0], number_units))
        else:
            gamma = gamma_from_init(
                J, num_reg1, ei_sign=ei_sign_init, mode=gamma_init,
                e_frac=e_frac, n_E=n_E_kw)
        gamma0 = gamma.copy()
        W = _init_W_from_J_gamma(
            J, gamma, num_reg1, zero_i_across=zero_i_across)
        J = j_from_dale_scalar(W, gamma, num_reg1, zero_i_across)
    elif intra_reparam:
        W = np.zeros_like(J)
        W[masks['intra_e']] = _inv_softplus(np.abs(J[masks['intra_e']]))
        W[masks['intra_i']] = _inv_softplus(np.abs(J[masks['intra_i']]))
        J[masks['intra_e']] = _softplus(W[masks['intra_e']])
        J[masks['intra_i']] = -_softplus(W[masks['intra_i']])
        J[masks['inter_i']] = 0.0

    J0 = J.copy()

    Adata = activity.copy()
    train_max = Adata.max()
    Adata = Adata / Adata.max()
    Adata = np.minimum(Adata, 0.999)
    Adata = np.maximum(Adata, -0.999)

    stdData = np.std(Adata[iTarget, :])

    iModelSample = numpy.zeros(len(tData), dtype=np.int32)
    for i in range(len(tData)):
        iModelSample[i] = (np.abs(tRNN - tData[i])).argmin()

    RNN = np.zeros((number_units, len(tRNN)))
    chi2s = []
    pVars = []
    PJ = P0 * np.eye(number_learn)
    local_r2 = ()
    reset_set = set(int(x) for x in np.asarray(resetPoints).ravel())

    if plotStatus is True:
        plt.rcParams.update({'font.size': 6})
        fig = plt.figure()
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        gs = GridSpec(nrows=2, ncols=4)
    else:
        fig = None
        gs = None

    for nRun in range(0, nRunTot):
        H = Adata[:, 0, np.newaxis]
        RNN[:, 0, np.newaxis] = nonLinearity(H)
        tLearn = 0
        iLearn = 0
        chi2 = 0.0
        dJ_acc = np.zeros((number_units, number_units), dtype=float)
        J_anchor = J.copy() if dale_scalar else None
        n_flip_run = 0

        for tt in range(1, len(tRNN)):
            tLearn += dtRNN
            if tt in reset_set:
                timepoint = math.floor(tt / dtFactor)
                H = Adata[:, timepoint]
                if dale_scalar:
                    J = j_from_dale_scalar(W, gamma, num_reg1, zero_i_across)
                else:
                    mask = J[:num_reg1, num_reg1:] < 0
                    J[:num_reg1, num_reg1:][mask] = 0
                    mask = J[num_reg1:, :num_reg1] < 0
                    J[num_reg1:, :num_reg1][mask] = 0
                    if zero_i_across:
                        J[masks['inter_i']] = 0.0
                if H.ndim == 1:
                    H = H[:, None]
            RNN[:, tt, np.newaxis] = nonLinearity(H)
            JR = (J.dot(RNN[:, tt]).reshape((number_units, 1)) +
                  inputWN[:, tt, np.newaxis])
            H = H + dtRNN * (-H + JR) / tauRNN
            if tLearn >= dtData:
                tLearn = 0
                err = RNN[:, tt, np.newaxis] - Adata[:, iLearn, np.newaxis]
                iLearn = iLearn + 1
                chi2 += np.mean(err ** 2)

                if nRun < nRunTrain:
                    r_slice = RNN[iTarget, tt].reshape(number_learn, 1)
                    k = PJ.dot(r_slice)
                    rPr = (r_slice).T.dot(k)[0, 0]
                    c = 1.0 / (1.0 + rPr)
                    PJ = PJ - c * (k.dot(k.T))
                    dJ = np.zeros_like(J)
                    dJ[:, iTarget.flatten()] = -c * np.outer(
                        err.flatten(), k.flatten())
                    if dale_scalar:
                        dJ_acc += dJ
                        W, gamma, J = dale_scalar_force_step(
                            W, gamma, dJ, num_reg1,
                            zero_i_across=zero_i_across,
                            gamma_gain=gamma_gain,
                            min_e_count=min_e_count,
                            gamma_l2=gamma_l2)
                        if (epoch_flip and flip_every is not None
                                and int(flip_every) > 0
                                and iLearn % int(flip_every) == 0):
                            W, gamma, J, nf = dale_scalar_reassign_from_unc(
                                W, gamma, J_anchor + dJ_acc, num_reg1,
                                zero_i_across=zero_i_across,
                                margin=flip_margin, dJ_acc=dJ_acc,
                                min_e_count=min_e_count)
                            n_flip_run += nf
                            J_anchor = J.copy()
                            dJ_acc[:] = 0.0
                    elif intra_reparam:
                        sigW = _sigmoid(W)
                        W[masks['intra_e']] = (
                            W[masks['intra_e']] + dJ[masks['intra_e']] * sigW[masks['intra_e']])
                        W[masks['intra_i']] = (
                            W[masks['intra_i']] + dJ[masks['intra_i']] * (-sigW[masks['intra_i']]))
                        J[masks['intra_e']] = _softplus(W[masks['intra_e']])
                        J[masks['intra_i']] = -_softplus(W[masks['intra_i']])
                        J[masks['inter_e']] = np.maximum(
                            J[masks['inter_e']] + dJ[masks['inter_e']], 0.0)
                        J[masks['inter_i']] = 0.0
                    else:
                        J[:, iTarget.flatten()] = (
                            J[:, iTarget.reshape((number_units))] +
                            dJ[:, iTarget.flatten()])
                        if zero_i_across:
                            J[masks['inter_i']] = 0.0

        if nRun < nRunTrain and dale_scalar and epoch_flip:
            W, gamma, J, nf = dale_scalar_reassign_from_unc(
                W, gamma, J_anchor + dJ_acc, num_reg1,
                zero_i_across=zero_i_across, margin=flip_margin,
                dJ_acc=dJ_acc, min_e_count=min_e_count)
            n_flip_run += nf
            J = j_from_dale_scalar(W, gamma, num_reg1, zero_i_across)

        # Dale-scalar E cells keep both local and across; Bio column-sparsify
        # would zero entire E-across columns (local XOR long-range).
        if nRun < nRunTrain and sparse_percent > 0 and not dale_scalar:
            percentile = percentile_list[nRun]
            print(percentile)
            if zero_i_across:
                if len(e2_rel):
                    temp = np.sum(J[:num_reg1, num_reg1:][:, e2_rel], axis=0)
                    low_indices, high_indices = get_lows(temp, percentile=percentile)
                    weak = e2_rel[low_indices[0]]
                    J[:num_reg1, num_reg1:][:, weak] = 0
                    local_r2 = (weak,)
                if len(e1_rel):
                    temp = np.sum(J[num_reg1:, :num_reg1][:, e1_rel], axis=0)
                    low_indices, high_indices = get_lows(temp, percentile=percentile)
                    weak = e1_rel[low_indices[0]]
                    J[num_reg1:, :num_reg1][:, weak] = 0
            else:
                temp = np.sum(J[:num_reg1, num_reg1:], axis=0)
                low_indices, high_indices = get_lows(temp, percentile=percentile)
                for idx in low_indices:
                    J[:num_reg1, num_reg1:][:, idx] = 0
                local_r2 = low_indices
                temp = np.sum(J[num_reg1:, :num_reg1], axis=0)
                low_indices, high_indices = get_lows(temp, percentile=percentile)
                for idx in low_indices:
                    J[num_reg1:, :num_reg1][:, idx] = 0
            if zero_i_across:
                J[masks['inter_i']] = 0.0

        rModelSample = RNN[iTarget, :][:, iModelSample]
        distance = np.linalg.norm(Adata[iTarget, :] - rModelSample)
        pVar = 1 - (distance / (math.sqrt(len(iTarget) * len(tData))
                    * stdData)) ** 2
        pVars.append(pVar)
        chi2s.append(chi2)
        if dale_scalar:
            n_E_hist.append(int(np.sum(gamma > 0)))
            pickle_agree_hist.append(float(np.mean(
                np.sign(gamma) == np.sign(ei_sign_init))))
            n_flip_hist.append(int(n_flip_run))
        if verbose:
            if dale_scalar:
                print('trial=%d pVar=%f chi2=%f nE=%d nI=%d pickle_agree=%.3f flips=%d' % (
                    nRun, pVar, chi2, n_E_hist[-1], int(np.sum(gamma < 0)),
                    pickle_agree_hist[-1], n_flip_run))
            else:
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

    if dale_scalar:
        J = j_from_dale_scalar(W, gamma, num_reg1, zero_i_across)
        ei_sign = np.where(gamma >= 0, 1.0, -1.0)
    elif zero_i_across:
        J[masks['inter_i']] = 0.0
        mask = J[:num_reg1, num_reg1:] < 0
        J[:num_reg1, num_reg1:][mask] = 0
        mask = J[num_reg1:, :num_reg1] < 0
        J[num_reg1:, :num_reg1][mask] = 0
    if intra_reparam:
        J[masks['intra_e']] = _softplus(W[masks['intra_e']])
        J[masks['intra_i']] = -_softplus(W[masks['intra_i']])
        J[masks['inter_i']] = 0.0

    out_params = {}
    out_params['dtFactor'] = dtFactor
    out_params['number_units'] = number_units
    out_params['g'] = g
    out_params['g_across'] = g_across
    out_params['g_loc'] = g_loc
    out_params['P0'] = P0
    out_params['tauRNN'] = tauRNN
    out_params['tauWN'] = tauWN
    out_params['ampInWN'] = ampInWN
    out_params['nRunTot'] = nRunTot
    out_params['nRunTrain'] = nRunTrain
    out_params['nRunFree'] = nRunFree
    out_params['nonLinearity'] = nonLinearity
    out_params['resetPoints'] = resetPoints
    out_params['sparse_percent'] = sparse_percent
    out_params['zero_i_across'] = bool(zero_i_across)
    out_params['intra_reparam'] = bool(intra_reparam)
    out_params['dale_scalar'] = bool(dale_scalar)
    out_params['gamma_gain'] = float(gamma_gain)
    out_params['gamma_l2'] = float(gamma_l2)
    out_params['epoch_flip'] = bool(epoch_flip)
    out_params['flip_every'] = None if flip_every is None else int(flip_every)
    out_params['flip_margin'] = float(flip_margin)

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
    out['W'] = W
    out['gamma'] = gamma
    out['gamma0'] = gamma0
    out['chi2s'] = chi2s
    out['pVars'] = pVars
    out['n_E_hist'] = n_E_hist
    out['pickle_agree_hist'] = pickle_agree_hist
    out['n_flip_hist'] = n_flip_hist
    out['stdData'] = stdData
    out['inputWN'] = inputWN
    out['iTarget'] = iTarget
    out['iNonTarget'] = iNonTarget
    out['params'] = out_params
    out['train_max'] = train_max
    out['local_r2'] = local_r2
    out['ei_sign'] = ei_sign
    out['ei_sign_init'] = ei_sign_init
    out['populations'] = populations
    return out


def pvar_corr_from_pred(Adata, pred, stdData=None):
    """pVar and Pearson corr for rates already sampled at data time."""
    Adata = np.asarray(Adata, dtype=float)
    pred = np.asarray(pred, dtype=float)
    T = min(Adata.shape[1], pred.shape[1])
    A = Adata[:, :T]
    P = pred[:, :T]
    if stdData is None:
        stdData = float(np.std(A))
    dist = np.linalg.norm(A - P)
    pvar = 1.0 - (dist / (math.sqrt(A.size) * (stdData + 1e-12))) ** 2
    corr = np.corrcoef(A.ravel(), P.ravel())[0, 1]
    sat = float(np.mean(np.abs(P) > 0.9))
    return {
        'pVar': float(pvar),
        'corr': float(corr) if np.isfinite(corr) else np.nan,
        'sat': sat,
        'pred_std': float(P.std()),
        'Adata_std': float(A.std()),
    }


def rollout_frozen_j(model, reset_every='train', n_trials=None, trial_length=None,
                     reuse_wn=True, seed=0):
    """Frozen-J Euler with a chosen teacher-force schedule.

    reset_every:
      'train' — the schedule the model was trained with
      None / 'trialstarts' — reset H at trial starts only
      int — trial starts plus every this many RNN steps (e.g. 25 = 100 ms)
      'none' — only the t=0 hidden state from data; no later resets
    """
    params = model['params']
    Adata = np.asarray(model['Adata'], dtype=float)
    J = np.asarray(model['J'], dtype=float)
    dtData = float(model.get('dtData', params.get('dtData', 0.02)))
    dtFactor = int(params['dtFactor'])
    dtRNN = float(model.get('dtRNN', dtData / float(dtFactor)))
    tauRNN = float(params['tauRNN'])
    nonLinearity = params['nonLinearity']
    number_units = int(params['number_units'])
    tData = np.asarray(model.get('tData', dtData * np.arange(Adata.shape[1])))
    tRNN = np.asarray(model.get('tRNN', np.arange(0, tData[-1] + dtRNN, dtRNN)))
    if n_trials is None or trial_length is None:
        n_trials, trial_length = _trial_shape(Adata.shape[1])
    if reset_every == 'train':
        resetPoints = np.asarray(params['resetPoints']).ravel().astype(np.int32)
    elif reset_every in (None, 'trialstarts'):
        resetPoints = make_reset_points(n_trials, trial_length, dtFactor, None)
    elif reset_every in ('none', 't0'):
        resetPoints = np.array([], dtype=np.int32)
    else:
        resetPoints = make_reset_points(
            n_trials, trial_length, dtFactor, int(reset_every))
    reset_set = set(int(x) for x in resetPoints)

    if reuse_wn and model.get('inputWN') is not None:
        inputWN = np.asarray(model['inputWN'], dtype=float)
        if inputWN.shape[1] < len(tRNN):
            reuse_wn = False
    if not (reuse_wn and model.get('inputWN') is not None):
        npr.seed(seed)
        tauWN = float(params.get('tauWN', 0.1))
        ampInWN = float(params.get('ampInWN', 0.001))
        ampWN = math.sqrt(tauWN / dtRNN)
        iWN = ampWN * npr.randn(number_units, len(tRNN))
        inputWN = np.ones((number_units, len(tRNN)))
        for tt in range(1, len(tRNN)):
            inputWN[:, tt] = iWN[:, tt] + (inputWN[:, tt - 1] - iWN[:, tt]) * np.exp(-(dtRNN / tauWN))
        inputWN = ampInWN * inputWN

    RNN = np.zeros((number_units, len(tRNN)))
    H = Adata[:, 0, np.newaxis]
    RNN[:, 0, np.newaxis] = nonLinearity(H)
    for tt in range(1, len(tRNN)):
        if tt in reset_set:
            timepoint = min(int(math.floor(tt / dtFactor)), Adata.shape[1] - 1)
            H = Adata[:, timepoint]
            if H.ndim == 1:
                H = H[:, None]
        RNN[:, tt, np.newaxis] = nonLinearity(H)
        JR = (J.dot(RNN[:, tt]).reshape((number_units, 1)) +
              inputWN[:, tt, np.newaxis])
        H = H + dtRNN * (-H + JR) / tauRNN

    i_model = np.array([(np.abs(tRNN - t)).argmin() for t in tData], dtype=int)
    i_model = np.clip(i_model, 0, RNN.shape[1] - 1)
    pred = RNN[:, i_model][:, :Adata.shape[1]]
    stats = pvar_corr_from_pred(Adata, pred, stdData=model.get('stdData'))
    stats['RNN'] = RNN
    stats['pred'] = pred
    stats['resetPoints'] = resetPoints
    stats['n_reset'] = int(len(resetPoints))
    stats['reset_every'] = reset_every
    return stats


def evaluate_teacher_force(model, n_trials=None, trial_length=None):
    """pVar/corr under the train schedule, 100 ms TF, trial starts, and t=0 only."""
    rows = []
    for label, re in (
            ('train_schedule', 'train'),
            ('reset25_100ms', 25),
            ('trialstarts', 'trialstarts'),
            ('t0_only', 'none')):
        st = rollout_frozen_j(model, reset_every=re,
                              n_trials=n_trials, trial_length=trial_length)
        rows.append(dict(
            schedule=label,
            reset_every=re,
            n_reset=st['n_reset'],
            pVar=st['pVar'],
            corr=st['corr'],
            sat=st['sat'],
            pred_std=st['pred_std'],
            pred=st['pred'],
            RNN=st['RNN'],
        ))
    return rows


def trainEIBioMultiRegionRNN(activity, dtData=1, dtFactor=1, g=1.5, tauRNN=0.01,
                        tauWN=0.1, ampInWN=0.01, nRunTrain=2000,
                        nRunFree=10, P0=1.0,
                        nonLinearity=np.tanh,
                        nonLinearity_inv=np.arctanh,
                        resetPoints=None,
                        plotStatus=True, verbose=True,
                        regions=None, g_across=None,
                        sparse_percent=80,
                        g_loc=(0, 0),
                        ei_sign=None,
                        populations=None,
                        adata_scale='z3',
                        target_radius=None,
                        max_radius=None,
                        project_every_update=True,
                        reset_state='rate',
                        align_error='prev',
                        init_intra='dale',
                        intra_dale='hard',
                        intra_dale_ramp=(0.3, 0.8),
                        project_intra_when='update',
                        max_radius_after_frac=None):
    """Dale-constrained two-region RNN on top of trainBioMultiRegionRNN.

    Inter-region weights are excitatory from E cells only (I→across = 0).
    Intra-region Dale (E >= 0, I <= 0) can be hard from the start, off
    (Bio-style signed intra), or annealed via intra_dale='anneal'.

    Fitted J and E/I currents are ground truth for this RNN, not recovered
    synapses of the recorded animal.
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
    if isinstance(g_loc, (int, float)):
        g_loc = (g_loc, g_loc)
    intra_dale = _normalize_intra_dale_mode(intra_dale)
    if init_intra not in ('dale', 'signed'):
        raise ValueError('Unknown init_intra: {}'.format(init_intra))
    if project_intra_when not in ('update', 'reset', 'epoch'):
        raise ValueError('Unknown project_intra_when: {}'.format(project_intra_when))
    if not project_every_update and project_intra_when == 'update':
        project_intra_when = 'reset'
    if intra_dale_ramp is None:
        intra_dale_ramp = (0.3, 0.8)

    number_units = activity.shape[0]
    number_learn = activity.shape[0]

    region1 = regions['region1']
    region2 = regions['region2']

    if populations is None:
        populations = {}
    if ei_sign is None:
        if not populations:
            raise ValueError('trainEIBioMultiRegionRNN requires ei_sign or populations')
        ei_sign = ei_sign_from_populations(populations, number_units)
    ei_sign = np.asarray(ei_sign, dtype=float).reshape(-1)
    if ei_sign.shape[0] != number_units:
        raise ValueError('ei_sign length {} != number_units {}'.format(
            ei_sign.shape[0], number_units))

    dtRNN = dtData / float(dtFactor)
    nRunTot = nRunTrain + nRunFree
    percentile_list = np.linspace(sparse_percent//2, sparse_percent, nRunTrain)
    print(percentile_list)

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

    masks = _dale_masks(number_units, num_reg1, ei_sign)
    e1_rel = np.where(ei_sign[:num_reg1] > 0)[0]
    e2_rel = np.where(ei_sign[num_reg1:] > 0)[0]
    J = _init_dale_J(number_units, num_reg1, ei_sign, g, g_across, g_loc,
                     init_intra=init_intra)
    _project_inter_inplace(J, masks)
    if intra_dale == 'hard':
        _project_intra_inplace(J, masks, alpha=1.0)
    rho0 = _spectral_radius(J)
    if target_radius is not None and rho0 > 0:
        J *= float(target_radius) / rho0
        _project_inter_inplace(J, masks)
        if intra_dale == 'hard':
            _project_intra_inplace(J, masks, alpha=1.0)
    J0 = J.copy()

    Adata, train_max = _prepare_adata(activity, adata_scale=adata_scale)

    stdData = np.std(Adata[iTarget, :])

    iModelSample = numpy.zeros(len(tData), dtype=np.int32)
    for i in range(len(tData)):
        iModelSample[i] = (np.abs(tRNN - tData[i])).argmin()

    RNN = np.zeros((number_units, len(tRNN)))
    chi2s = []
    pVars = []
    dale_alphas = []
    illegal_intra = []
    rhos = []
    PJ = P0*np.eye(number_learn)
    local_r2 = (np.array([], dtype=int),)
    reset_set = set(int(x) for x in np.atleast_1d(resetPoints))

    if plotStatus is True:
        plt.rcParams.update({'font.size': 6})
        fig = plt.figure()
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        gs = GridSpec(nrows=2, ncols=4)
    else:
        fig = None

    for nRun in range(0, nRunTot):
        alpha = _intra_dale_alpha(nRun, nRunTrain, intra_dale, intra_dale_ramp)
        cap = _radius_cap_this_run(nRun, nRunTrain, max_radius, max_radius_after_frac)
        intra_at_reset = project_intra_when in ('reset', 'update')
        intra_at_update = project_intra_when == 'update'
        H = _hidden_from_rates(Adata[:, 0], reset_state, nonLinearity_inv)
        RNN[:, 0, np.newaxis] = nonLinearity(H)
        tLearn = 0
        iLearn = 0
        chi2 = 0.0

        for tt in range(1, len(tRNN)):
            tLearn += dtRNN
            if tt in reset_set:
                timepoint = math.floor(tt / dtFactor)
                if timepoint >= Adata.shape[1]:
                    timepoint = Adata.shape[1] - 1
                H = _hidden_from_rates(Adata[:, timepoint], reset_state, nonLinearity_inv)
                iLearn = int(timepoint)
                _project_inter_inplace(J, masks)
                if intra_at_reset:
                    _project_intra_inplace(J, masks, alpha=alpha)
                _maybe_cap_radius(J, masks, cap, alpha)
                if H.ndim==1:
                    H = H[:,None]
            RNN[:, tt, np.newaxis] = nonLinearity(H)
            JR = (J.dot(RNN[:, tt]).reshape((number_units, 1)) +
                  inputWN[:, tt, np.newaxis])
            H = H + dtRNN*(-H + JR)/tauRNN
            if tLearn >= dtData:
                tLearn = 0
                if align_error == 'current':
                    data_idx = int(min(math.floor(tt / float(dtFactor)), Adata.shape[1] - 1))
                    err = RNN[:, tt, np.newaxis] - Adata[:, data_idx, np.newaxis]
                    iLearn = data_idx + 1
                else:
                    if iLearn >= Adata.shape[1]:
                        continue
                    err = RNN[:, tt, np.newaxis] - Adata[:, iLearn, np.newaxis]
                    iLearn = iLearn + 1
                chi2 += np.mean(err ** 2)

                if nRun < nRunTrain:
                    r_slice = RNN[iTarget, tt].reshape(number_learn, 1)
                    k = PJ.dot(r_slice)
                    rPr = (r_slice).T.dot(k)[0, 0]
                    c = 1.0/(1.0 + rPr)
                    PJ = PJ - c*(k.dot(k.T))
                    J[:, iTarget.flatten()] = J[:, iTarget.reshape((number_units))] - c*np.outer(err.flatten(), k.flatten())
                    J[masks['inter_i']] = 0.0
                    if project_every_update:
                        _project_inter_inplace(J, masks)
                    if intra_at_update:
                        _project_intra_inplace(J, masks, alpha=alpha)

        if nRun < nRunTrain and sparse_percent > 0:
            percentile=percentile_list[nRun]
            print(percentile)
            if len(e2_rel):
                temp = np.sum(J[:num_reg1, num_reg1:][:, e2_rel], axis=0)
                low_indices, high_indices = get_lows(temp, percentile=percentile)
                weak = e2_rel[low_indices[0]]
                J[:num_reg1, num_reg1:][:, weak] = 0
                local_r2 = (weak,)
            if len(e1_rel):
                temp = np.sum(J[num_reg1:, :num_reg1][:, e1_rel], axis=0)
                low_indices, high_indices = get_lows(temp, percentile=percentile)
                weak = e1_rel[low_indices[0]]
                J[num_reg1:, :num_reg1][:, weak] = 0
            _project_inter_inplace(J, masks)
            _project_intra_inplace(J, masks, alpha=alpha)

        if nRun < nRunTrain:
            _project_inter_inplace(J, masks)
            _project_intra_inplace(J, masks, alpha=alpha)
            _maybe_cap_radius(J, masks, cap, alpha)

        rModelSample = RNN[iTarget, :][:, iModelSample]
        distance = np.linalg.norm(Adata[iTarget, :] - rModelSample)
        pVar = 1 - (distance / (math.sqrt(len(iTarget) * len(tData))
                    * stdData)) ** 2
        pVars.append(pVar)
        chi2s.append(chi2)
        dale_alphas.append(float(alpha))
        illegal_intra.append(_illegal_intra_mass(J, masks))
        rhos.append(_spectral_radius(J))
        if verbose:
            print('trial=%d pVar=%f chi2=%f alpha=%.3f illegal=%.4g rho=%.3f' % (
                nRun, pVar, chi2, alpha, illegal_intra[-1], rhos[-1]))
        if intra_dale == 'anneal' and (nRun + 1) == nRunTrain:
            _project_intra_inplace(J, masks, alpha=1.0)
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
    out_params['g_across'] = g_across
    out_params['g_loc'] = g_loc
    out_params['P0'] = P0
    out_params['tauRNN'] = tauRNN
    out_params['tauWN'] = tauWN
    out_params['ampInWN'] = ampInWN
    out_params['nRunTot'] = nRunTot
    out_params['nRunTrain'] = nRunTrain
    out_params['nRunFree'] = nRunFree
    out_params['nonLinearity'] = nonLinearity
    out_params['resetPoints'] = resetPoints
    out_params['sparse_percent'] = sparse_percent
    out_params['dale'] = True
    out_params['adata_scale'] = adata_scale
    out_params['target_radius'] = target_radius
    out_params['max_radius'] = max_radius
    out_params['project_every_update'] = project_every_update
    out_params['reset_state'] = reset_state
    out_params['align_error'] = align_error
    out_params['init_intra'] = init_intra
    out_params['intra_dale'] = intra_dale
    out_params['intra_dale_ramp'] = tuple(intra_dale_ramp)
    out_params['project_intra_when'] = project_intra_when
    out_params['max_radius_after_frac'] = max_radius_after_frac
    out_params['rho0'] = rho0
    out_params['rho_init'] = _spectral_radius(J0)

    out = {}
    out['regions'] = regions
    out['populations'] = populations
    out['ei_sign'] = ei_sign
    out['dale'] = True
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
    out['dale_alphas'] = dale_alphas
    out['illegal_intra'] = illegal_intra
    out['rhos'] = rhos
    out['stdData'] = stdData
    out['inputWN'] = inputWN
    out['iTarget'] = iTarget
    out['iNonTarget'] = iNonTarget
    out['params'] = out_params
    out['train_max'] = train_max
    out['local_r2'] = local_r2

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
    H = Adata[:, np.random.choice(Adata.shape[1])]
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


def resolve_opto_target_population(model):
    """Dataset yaml label (CFA_I or RFA_I), not the old J-sum / region2 heuristic."""
    name = model.get('opto_target_population')
    if name:
        return str(name)
    pkl_path = model.get('pkl_path')
    if pkl_path:
        pkl_path, yaml_path, snapshot_path = _resolve_ei_dataset_paths(pkl_path)
        meta = _opto_fields_from_cfgs(
            _load_yaml_dict(yaml_path), _load_yaml_dict(snapshot_path), pkl_path)
        if meta.get('opto_target_population'):
            return str(meta['opto_target_population'])
    pops = model.get('populations') or {}
    for fallback in ('RFA_I', 'CFA_I'):
        if fallback in pops and len(np.asarray(pops[fallback])):
            return fallback
    raise ValueError('Could not resolve opto_target_population from model or dataset yaml')


def opto_target_indices(model, target_population=None):
    """Unit indices for the dataset's stimulated I population."""
    name = target_population or resolve_opto_target_population(model)
    pops = model.get('populations') or {}
    if name not in pops:
        raise KeyError('opto target {} not in populations {}'.format(
            name, list(pops.keys())))
    idx = np.asarray(pops[name], dtype=int)
    if idx.size == 0:
        raise ValueError('opto target population {} is empty'.format(name))
    return name, idx


def _opto_inhib_pulse(n_units, n_times, target_idx, stim_mask, optoAmp):
    """Existing inhib_only protocol: +half-normal current on target units while on."""
    optoInp = np.zeros((n_units, n_times))
    on = np.where(np.asarray(stim_mask, dtype=bool))[0]
    target_idx = np.asarray(target_idx, dtype=int)
    n_tgt = int(target_idx.size)
    if n_tgt == 0 or on.size == 0:
        return optoInp
    for i in on:
        optoInp[target_idx, i] = truncnorm.rvs(
            a=0, b=np.inf, loc=0, scale=1, size=n_tgt)
    return float(optoAmp) * optoInp


def _trial_stim_mask(n_times, n_trials, trial_rnn, dtRNN, stim_onset_s, dur):
    """25 ms (default) pulse on every trial, starting at the dataset pre window."""
    mask = np.zeros(n_times, dtype=bool)
    onset_steps = int(stim_onset_s / dtRNN)
    dur_steps = max(1, int(dur / dtRNN))
    for tr in range(int(n_trials)):
        start = int(tr) * int(trial_rnn) + onset_steps
        stop = min(start + dur_steps, n_times)
        if start < n_times:
            mask[max(start, 0):stop] = True
    return mask


def simulate_pseudo_opto_trials(model, n_trials=None, trial_length=None,
                                target_population=None, dur=0.025, optoAmp=None,
                                stim_onset_s=None, seed=0, reuse_wn=True,
                                with_control=True):
    """Trial-start rollout with the existing I-cell opto pulse on the dataset target.

    Does not change the stim waveform (positive truncated-normal, 25 ms, optoAmp).
    Control and opto share the same white noise and trial-start resets from Adata.
    """
    params = model['params']
    Adata = np.asarray(model['Adata'], dtype=float)
    J = np.asarray(model['J'], dtype=float)
    dtData = float(model.get('dtData', params.get('dtData', 0.02)))
    dtFactor = int(params['dtFactor'])
    dtRNN = float(model.get('dtRNN', dtData / float(dtFactor)))
    tauRNN = float(params['tauRNN'])
    nonLinearity = params['nonLinearity']
    number_units = int(params['number_units'])
    if n_trials is None:
        n_trials = model.get('n_trials')
    if trial_length is None:
        trial_length = model.get('trial_length')
    if n_trials is None or trial_length is None:
        n_trials, trial_length = _trial_shape(
            Adata.shape[1], trial_length=trial_length)
    n_trials = int(n_trials)
    trial_length = int(trial_length)
    n_data = n_trials * trial_length
    if Adata.shape[1] < n_data:
        n_trials = Adata.shape[1] // trial_length
        n_data = n_trials * trial_length
    Adata = Adata[:, :n_data]

    target_name, target_idx = opto_target_indices(model, target_population)
    if optoAmp is None:
        optoAmp = params.get('ampInWN', 0.001)
    if stim_onset_s is None:
        stim_onset_s = model.get('stim_onset_s', 0.02)

    trial_rnn = trial_length * dtFactor
    n_rnn = n_trials * trial_rnn
    tRNN = np.arange(n_rnn, dtype=float) * dtRNN
    tData = dtData * np.arange(n_data)
    resetPoints = make_reset_points(n_trials, trial_length, dtFactor, None)
    reset_set = set(int(x) for x in resetPoints)

    if reuse_wn and model.get('inputWN') is not None:
        inputWN = np.asarray(model['inputWN'], dtype=float)[:, :n_rnn]
        if inputWN.shape[1] < n_rnn:
            reuse_wn = False
    if not (reuse_wn and model.get('inputWN') is not None):
        npr.seed(seed)
        tauWN = float(params.get('tauWN', 0.1))
        ampInWN = float(params.get('ampInWN', 0.001))
        ampWN = math.sqrt(tauWN / dtRNN)
        iWN = ampWN * npr.randn(number_units, n_rnn)
        inputWN = np.ones((number_units, n_rnn))
        for tt in range(1, n_rnn):
            inputWN[:, tt] = iWN[:, tt] + (inputWN[:, tt - 1] - iWN[:, tt]) * np.exp(-(dtRNN / tauWN))
        inputWN = ampInWN * inputWN

    stim_mask = _trial_stim_mask(
        n_rnn, n_trials, trial_rnn, dtRNN, float(stim_onset_s), float(dur))
    npr.seed(seed + 1)
    optoInp = _opto_inhib_pulse(number_units, n_rnn, target_idx, stim_mask, optoAmp)

    reset_state = params.get('reset_state', 'rate')
    nonLinearity_inv = params.get('nonLinearity_inv', np.arctanh)

    def _rollout(extra_input):
        RNN = np.zeros((number_units, n_rnn))
        H = _hidden_from_rates(Adata[:, 0], reset_state, nonLinearity_inv)
        if H.ndim == 1:
            H = H[:, None]
        RNN[:, 0, np.newaxis] = nonLinearity(H)
        for tt in range(1, n_rnn):
            if tt in reset_set:
                timepoint = min(int(math.floor(tt / dtFactor)), Adata.shape[1] - 1)
                H = _hidden_from_rates(Adata[:, timepoint], reset_state, nonLinearity_inv)
                if H.ndim == 1:
                    H = H[:, None]
            RNN[:, tt, np.newaxis] = nonLinearity(H)
            JR = (J.dot(RNN[:, tt]).reshape((number_units, 1))
                  + inputWN[:, tt, np.newaxis]
                  + extra_input[:, tt, np.newaxis])
            H = H + dtRNN * (-H + JR) / tauRNN
        return RNN

    RNN_opto = _rollout(optoInp)
    RNN_ctrl = _rollout(np.zeros_like(optoInp)) if with_control else None

    i_model = np.array([(np.abs(tRNN - t)).argmin() for t in tData], dtype=int)
    i_model = np.clip(i_model, 0, n_rnn - 1)
    pred_opto = RNN_opto[:, i_model]
    pred_ctrl = None if RNN_ctrl is None else RNN_ctrl[:, i_model]
    stim_mask_data = stim_mask[i_model]
    return {
        'target_population': target_name,
        'target_idx': target_idx,
        'n_trials': n_trials,
        'trial_length': trial_length,
        'dur': float(dur),
        'optoAmp': float(optoAmp),
        'stim_onset_s': float(stim_onset_s),
        'stim_mask': stim_mask,
        'stim_mask_data': stim_mask_data,
        'RNN_ctrl': RNN_ctrl,
        'RNN_opto': RNN_opto,
        'pred_ctrl': pred_ctrl,
        'pred_opto': pred_opto,
        'Adata': Adata,
    }


def plotFit(model):
    pVars = model['pVars']
    RNN = model['RNN']
    tRNN = model['tRNN']
    dtFactor = model['params']['dtFactor']
    Adata=model['Adata']
    iTarget = model['iTarget']
    tData = model['tData']
    chi2s = model['chi2s']

    try:
        r2 = weighted_r2(Adata.T, RNN[:, ::dtFactor].T)
    except NameError:
        pred = RNN[:, ::dtFactor][:, :Adata.shape[1]]
        r2 = float(np.mean(_r2_per_neuron(Adata, pred)))
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


def check_dale_constraints(model, atol=1e-12):
    """Verify intra Dale signs, E-only non-negative inter-region, I→across = 0."""
    J = np.asarray(model['J'])
    ei_sign = np.asarray(model['ei_sign']).reshape(-1)
    num_reg1 = len(model['regions']['region1'])
    masks = _dale_masks(J.shape[0], num_reg1, ei_sign)
    intra_e_min = float(J[masks['intra_e']].min()) if np.any(masks['intra_e']) else 0.0
    intra_i_max = float(J[masks['intra_i']].max()) if np.any(masks['intra_i']) else 0.0
    inter_e_min = float(J[masks['inter_e']].min()) if np.any(masks['inter_e']) else 0.0
    inter_i_abs = float(np.max(np.abs(J[masks['inter_i']]))) if np.any(masks['inter_i']) else 0.0
    ok = {
        'intra_e_nonneg': intra_e_min >= -atol,
        'intra_i_nonpos': intra_i_max <= atol,
        'inter_e_nonneg': inter_e_min >= -atol,
        'inter_i_zero': inter_i_abs <= atol,
    }
    ok['all'] = all(ok.values())
    return {
        'ok': ok,
        'intra_e_min': intra_e_min,
        'intra_i_max': intra_i_max,
        'inter_e_min': inter_e_min,
        'inter_i_absmax': inter_i_abs,
    }


def population_currents(J, rates, populations, regions, tauRNN):
    """Source-population currents into CFA and RFA: J[trg, src] @ r_src / tau."""
    currents = {}
    for src_name, src_idx in populations.items():
        src_idx = np.asarray(src_idx)
        r_src = rates[src_idx, :]
        for trg_name, trg_key in (('CFA', 'region1'), ('RFA', 'region2')):
            trg_idx = np.asarray(regions[trg_key])
            currents['{}_to_{}'.format(src_name, trg_name)] = (
                J[np.ix_(trg_idx, src_idx)].dot(r_src) / float(tauRNN))
    return currents


def _rates_to_poisson_spikes(rates, scaler, train_max, poisson_mult=5):
    """Inverse-scale tanh rates to spike counts (N x T), matching CurbdModel.rates2spikes."""
    rates_tn = np.asarray(rates).T
    scaled = scaler.inverse_transform(rates_tn * train_max) * poisson_mult
    scaled = np.maximum(scaled, 0.0)
    spikes_tn = poisson.rvs(mu=scaled, size=scaled.shape)
    return spikes_tn.T


def export_ei_groundtruth(model, sim_rates=None, t=None, poisson_spikes=False,
                          poisson_mult=5):
    """Export simulated rates/currents as model ground truth after fitting.

    Labels and currents are ground truth for this RNN, not biological synapses.
    If sim_rates is omitted, autonomous rates are generated with simulate().
    """
    populations = model['populations']
    regions = model['regions']
    tauRNN = model['params']['tauRNN']
    J = np.asarray(model['J'])

    if sim_rates is None:
        if t is None:
            t = float(model['tData'][-1]) if model.get('tData') is not None else 1.0
            t = max(t, 0.5)
        sim_rates = simulate(model, t)
    sim_rates = np.asarray(sim_rates)

    rate_splits = {name: sim_rates[np.asarray(idx), :] for name, idx in populations.items()}
    currents = population_currents(J, sim_rates, populations, regions, tauRNN)

    out = {
        'model': model,
        'rates': sim_rates,
        'rate_splits': rate_splits,
        'currents': currents,
        'labels': {
            'ei_sign': np.asarray(model['ei_sign']),
            'populations': populations,
            'regions': regions,
        },
        'note': (
            'J, ei_sign, and currents are ground truth for this fitted RNN, '
            'not recovered biological synapses. Identifiability is unchanged: '
            'many Dale-feasible J matrices can match the same rates.'
        ),
    }
    if poisson_spikes:
        scaler = model.get('scaler')
        train_max = model.get('train_max')
        if scaler is None or train_max is None:
            raise ValueError('poisson_spikes requires model["scaler"] and model["train_max"]')
        spikes = _rates_to_poisson_spikes(sim_rates, scaler, train_max, poisson_mult)
        out['spikes'] = spikes
        out['spike_splits'] = {name: spikes[np.asarray(idx), :]
                               for name, idx in populations.items()}
    return out


POP_COLORS = {
    'CFA_E': '#d62728',
    'CFA_I': '#1f77b4',
    'RFA_E': '#ff7f0e',
    'RFA_I': '#2ca02c',
}


def load_ei_fit(path):
    """Load a pickle from trainEIBioFromDatasets.py (or a raw model dict)."""
    with open(path, 'rb') as f:
        blob = pickle.load(f)
    if isinstance(blob, dict) and 'J' in blob:
        return blob, None
    if isinstance(blob, dict) and 'model' in blob:
        return blob['model'], blob.get('ground_truth')
    raise ValueError('Unrecognized fit pickle: {}'.format(path))


def _ordered_populations(populations):
    items = []
    for name, idx in populations.items():
        idx = np.asarray(idx)
        items.append((int(idx.min()), int(idx.max()) + 1, name, idx))
    items.sort(key=lambda x: x[0])
    return items


def _unit_population_name(unit, populations):
    for name, idx in populations.items():
        if unit in np.asarray(idx):
            return name
    return '?'


def model_rates_at_data(model):
    """RNN rates sampled at the data timestamps (N x T_data)."""
    Adata = np.asarray(model['Adata'])
    RNN = np.asarray(model['RNN'])
    tRNN = np.asarray(model['tRNN'])
    tData = np.asarray(model['tData'])
    i_model = np.array([(np.abs(tRNN - t)).argmin() for t in tData], dtype=int)
    i_model = np.clip(i_model, 0, RNN.shape[1] - 1)
    pred = RNN[:, i_model]
    T = min(pred.shape[1], Adata.shape[1])
    return Adata[:, :T], pred[:, :T]


def _r2_per_neuron(true, pred):
    ss_res = np.sum((true - pred) ** 2, axis=1)
    mu = true.mean(axis=1, keepdims=True)
    ss_tot = np.sum((true - mu) ** 2, axis=1)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def summarize_ei_fit(model):
    """Convergence and rate-match metrics. pVar should rise toward 1 (usable > 0.5)."""
    Adata, pred = model_rates_at_data(model)
    pVars = np.asarray(model.get('pVars', []), dtype=float)
    chi2s = np.asarray(model.get('chi2s', []), dtype=float)
    r2 = _r2_per_neuron(Adata, pred)
    corr = np.corrcoef(Adata.ravel(), pred.ravel())[0, 1]
    rmse = float(np.sqrt(np.mean((Adata - pred) ** 2)))
    sat_model = float(np.mean(np.abs(pred) > 0.9))
    sat_data = float(np.mean(np.abs(Adata) > 0.9))
    spec_rad = float(np.max(np.abs(np.linalg.eigvals(np.asarray(model['J'])))))
    pvar_final = float(pVars[-1]) if len(pVars) else np.nan
    if len(pVars) >= 2:
        improving = bool(pVars[-1] > pVars[0] + 1e-6)
    else:
        improving = False
    if pvar_final >= 0.5 and sat_model < 0.5:
        status = 'PASS'
    elif pvar_final >= 0.0 and improving:
        status = 'PARTIAL'
    else:
        status = 'FAIL'
    return {
        'status': status,
        'pVars': pVars,
        'chi2s': chi2s,
        'pvar_final': pvar_final,
        'chi2_final': float(chi2s[-1]) if len(chi2s) else np.nan,
        'improving': improving,
        'rmse': rmse,
        'corr': float(corr) if np.isfinite(corr) else np.nan,
        'r2_mean': float(np.mean(r2)),
        'r2_median': float(np.median(r2)),
        'r2_frac_pos': float(np.mean(r2 > 0)),
        'r2_frac_half': float(np.mean(r2 > 0.5)),
        'r2_per_neuron': r2,
        'sat_model': sat_model,
        'sat_data': sat_data,
        'spectral_radius': spec_rad,
        'Adata_std': float(Adata.std()),
        'pred_std': float(pred.std()),
        'train_max': float(model.get('train_max', np.nan)),
        'dale_alpha_final': float(model['dale_alphas'][-1]) if model.get('dale_alphas') else np.nan,
        'illegal_intra_final': float(model['illegal_intra'][-1]) if model.get('illegal_intra') else np.nan,
    }


def print_ei_fit_report(model, metrics=None):
    if metrics is None:
        metrics = summarize_ei_fit(model)
    dale = check_dale_constraints(model)
    print('=== EI CURBD fit report ===')
    print('convergence: {}  (pVar should rise toward 1; >0.5 is a usable match)'.format(
        metrics['status']))
    print('  pVar: {} -> {:.4f}  improving={}'.format(
        ' -> '.join('{:.3f}'.format(x) for x in metrics['pVars'][:3]),
        metrics['pvar_final'], metrics['improving']))
    print('  chi2 final={:.4g}  (should fall across training runs)'.format(
        metrics['chi2_final']))
    print('rate match: corr={:.4f}  RMSE={:.4g}  mean neuron R2={:.4g}  '
          'median R2={:.4g}  frac R2>0={:.3f}  frac R2>0.5={:.3f}'.format(
              metrics['corr'], metrics['rmse'], metrics['r2_mean'],
              metrics['r2_median'], metrics['r2_frac_pos'], metrics['r2_frac_half']))
    print('  data std={:.4g}  model std={:.4g}  train_max={:.4g}'.format(
        metrics['Adata_std'], metrics['pred_std'], metrics['train_max']))
    print('  |rate|>0.9: data={:.2%}  model={:.2%}  (tanh saturation)'.format(
        metrics['sat_data'], metrics['sat_model']))
    print('  spectral radius of J={:.3g}  (trained CURBD is typically O(1))'.format(
        metrics['spectral_radius']))
    print('Dale: ok={}  intra E min={:.4g}  intra I max={:.4g}  '
          'inter E min={:.4g}  I->across absmax={:.4g}'.format(
              dale['ok']['all'], dale['intra_e_min'], dale['intra_i_max'],
              dale['inter_e_min'], dale['inter_i_absmax']))
    if model.get('dale_alphas') is not None:
        print('  intra Dale alpha final={:.3f}  illegal intra L2={:.4g}'.format(
            metrics.get('dale_alpha_final', float('nan')),
            metrics.get('illegal_intra_final', float('nan'))))
    if metrics['status'] != 'PASS':
        print('If pVar is largely negative and model rates hug +/-1, FORCE is not '
              'tracking the data (often Adata was compressed by /train_max, or '
              'nRunTrain is too small). Re-fit before treating J as ground truth.')
    return metrics


def _j_init_blocks(J, model):
    """Split J into local/cross x E/I (and CFA vs RFA) source blocks."""
    J = np.asarray(J)
    ei = np.asarray(model['ei_sign']).reshape(-1)
    n1 = len(model['regions']['region1'])
    e = ei > 0
    i = ei < 0
    return {
        'local_CFA_E': J[:n1, :n1][:, e[:n1]].ravel(),
        'local_CFA_I': J[:n1, :n1][:, i[:n1]].ravel(),
        'local_RFA_E': J[n1:, n1:][:, e[n1:]].ravel(),
        'local_RFA_I': J[n1:, n1:][:, i[n1:]].ravel(),
        'cross_CFA_E_to_RFA': J[n1:, :n1][:, e[:n1]].ravel(),
        'cross_CFA_I_to_RFA': J[n1:, :n1][:, i[:n1]].ravel(),
        'cross_RFA_E_to_CFA': J[:n1, n1:][:, e[n1:]].ravel(),
        'cross_RFA_I_to_CFA': J[:n1, n1:][:, i[n1:]].ravel(),
    }


def _halfnormal_pdf(x, scale):
    scale = float(scale)
    if scale <= 0:
        return np.zeros_like(x, dtype=float)
    return np.sqrt(2.0 / np.pi) / scale * np.exp(-0.5 * (x / scale) ** 2)


def plot_init_distributions(model, which='J0'):
    """Histograms of E/I weights at init (J0) or after training (J).

    Local = intra-region columns. Cross = inter-region. I->across should be 0.
    Dashed curves are the intended init PDFs (folded Gaussian / truncnorm).
    """
    if which == 'J0':
        if model.get('J0') is None:
            raise ValueError('model has no J0; pass which="J" for trained weights')
        J = np.asarray(model['J0'])
        title_prefix = 'Init J0'
    else:
        J = np.asarray(model['J'])
        title_prefix = 'Trained J'
    blocks = _j_init_blocks(J, model)
    params = model.get('params', {})
    n = int(params.get('number_units', J.shape[0]))
    g = float(params.get('g', 1.5))
    g_across = float(params.get('g_across', g))
    g_loc = params.get('g_loc', (0.0, 0.0))
    if isinstance(g_loc, (int, float)):
        g_loc = (g_loc, g_loc)
    inv_sqrt = 1.0 / math.sqrt(n)
    s_local = g * inv_sqrt
    loc_cfa = float(g_loc[0]) * inv_sqrt
    loc_rfa = float(g_loc[1]) * inv_sqrt
    s_cross = (g_across / g) * inv_sqrt

    panels = [
        ('local_CFA_E', 'Local CFA E', POP_COLORS['CFA_E'], 'e', loc_cfa, s_local),
        ('local_CFA_I', 'Local CFA I', POP_COLORS['CFA_I'], 'i', loc_cfa, s_local),
        ('local_RFA_E', 'Local RFA E', POP_COLORS['RFA_E'], 'e', loc_rfa, s_local),
        ('local_RFA_I', 'Local RFA I', POP_COLORS['RFA_I'], 'i', loc_rfa, s_local),
        ('cross_CFA_E_to_RFA', 'Cross CFA_E -> RFA', POP_COLORS['CFA_E'], 'e_cross', 0.0, s_cross),
        ('cross_CFA_I_to_RFA', 'Cross CFA_I -> RFA (must be 0)', POP_COLORS['CFA_I'], 'zero', 0.0, 0.0),
        ('cross_RFA_E_to_CFA', 'Cross RFA_E -> CFA', POP_COLORS['RFA_E'], 'e_cross', 0.0, s_cross),
        ('cross_RFA_I_to_CFA', 'Cross RFA_I -> CFA (must be 0)', POP_COLORS['RFA_I'], 'zero', 0.0, 0.0),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.2))
    print('=== {} weight distributions ==='.format(title_prefix))
    print('{:<28} {:>7} {:>10} {:>10} {:>10} {:>8}'.format(
        'block', 'n', 'mean', 'std', 'frac0', 'minmax'))
    for ax, (key, label, color, kind, loc, scale) in zip(axes.ravel(), panels):
        w = np.asarray(blocks[key], dtype=float)
        frac0 = float(np.mean(w == 0)) if w.size else np.nan
        print('{:<28} {:>7d} {:>10.4g} {:>10.4g} {:>10.3f} [{:.3g}, {:.3g}]'.format(
            key, int(w.size), float(w.mean()) if w.size else np.nan,
            float(w.std()) if w.size else np.nan, frac0,
            float(w.min()) if w.size else np.nan,
            float(w.max()) if w.size else np.nan))
        if w.size == 0:
            ax.set_title(label, fontsize=9)
            continue
        if kind == 'zero' or np.allclose(w, 0):
            ax.bar([0.0], [w.size], width=0.02, color=color, alpha=0.7)
            ax.set_xlim(-0.15, 0.15)
        else:
            ax.hist(w, bins=40, density=True, color=color, alpha=0.7, edgecolor='none')
            xs = np.linspace(w.min() - 0.02, w.max() + 0.02, 400)
            if kind == 'e':
                # g*|N(0,1)| + g_loc, then /sqrt(N), clip >= 0
                pdf = _halfnormal_pdf(np.maximum(xs - loc, 0.0), scale)
                pdf[(xs < max(loc, 0.0))] = 0.0
                ax.plot(xs, pdf, 'k--', lw=1.0, label='folded N(0,g)/sqrt(N)')
            elif kind == 'i':
                pdf = _halfnormal_pdf(np.maximum(loc - xs, 0.0), scale)
                pdf[(xs > min(loc, 0.0))] = 0.0
                ax.plot(xs, pdf, 'k--', lw=1.0, label='-folded N(0,g)/sqrt(N)')
            elif kind == 'e_cross':
                pdf = _halfnormal_pdf(np.maximum(xs, 0.0), scale)
                pdf[xs < 0] = 0.0
                ax.plot(xs, pdf, 'k--', lw=1.0, label='truncnorm /sqrt(N)')
            ax.legend(fontsize=6, loc='upper right')
        ax.axvline(0, color='k', lw=0.5)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('weight')
        ax.set_ylabel('density')
    fig.suptitle('{}  (local = intra-region, cross = inter-region)'.format(title_prefix))
    fig.tight_layout()
    return fig, axes


def plot_dale_J(model, ax=None):
    """Heatmap of J. Columns = source (presynaptic), rows = readout/target."""
    J = np.asarray(model['J'])
    pops = _ordered_populations(model['populations'])
    vmax = np.percentile(np.abs(J), 99) or 1.0
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
    else:
        fig = ax.figure
    im = ax.imshow(J, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto',
                   interpolation='nearest')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='J (source -> readout)')
    ticks = []
    ticklabels = []
    for start, end, name, _idx in pops:
        if start > 0:
            ax.axhline(start - 0.5, color='k', lw=0.8)
            ax.axvline(start - 0.5, color='k', lw=0.8)
        ticks.append(0.5 * (start + end - 1))
        ticklabels.append('{}\n{}-{}'.format(name, start, end - 1))
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, fontsize=8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels, fontsize=8)
    ax.set_xlabel('source neuron (presynaptic column)')
    ax.set_ylabel('readout neuron (postsynaptic row)')
    ax.set_title('Directed interaction matrix J')
    if created:
        fig.tight_layout()
    return fig, ax


def plot_ei_source_columns(model, n_per_pop=1, ax=None):
    """Outgoing J columns for example E/I sources, colored by readout population."""
    J = np.asarray(model['J'])
    pops = _ordered_populations(model['populations'])
    chosen = []
    for _start, _end, name, idx in pops:
        if len(idx) == 0:
            continue
        strength = np.sum(np.abs(J[:, idx]), axis=0)
        order = np.argsort(strength)[::-1]
        for k in range(min(n_per_pop, len(idx))):
            chosen.append((name, int(idx[order[k]]), float(strength[order[k]])))
    n = max(len(chosen), 1)
    created = ax is None
    if created:
        fig, axes = plt.subplots(n, 1, figsize=(8, 2.2 * n), sharex=True)
        if n == 1:
            axes = [axes]
    else:
        fig = ax.figure
        axes = [ax]
    colors = np.empty(J.shape[0], dtype=object)
    colors[:] = '#888888'
    for _start, _end, name, idx in pops:
        colors[idx] = POP_COLORS.get(name, '#888888')
    x = np.arange(J.shape[0])
    for ax_i, (src_name, src, _s) in zip(axes, chosen):
        w = J[:, src]
        ax_i.bar(x, w, color=colors, width=1.0, linewidth=0)
        for start, _end, name, _idx in pops:
            if start > 0:
                ax_i.axvline(start - 0.5, color='k', lw=0.6)
        ax_i.axhline(0, color='k', lw=0.5)
        ax_i.set_ylabel('J[:, {}]'.format(src))
        ax_i.set_title('source {} unit {} (readout neurons on x; E red/orange, I blue/green)'.format(
            src_name, src), fontsize=9)
    axes[-1].set_xlabel('readout neuron index')
    ticks = [0.5 * (s + e - 1) for s, e, _n, _i in pops]
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([n for _s, _e, n, _i in pops])
    if created:
        fig.tight_layout()
    return fig, axes


def plot_convergence(model, ax=None):
    pVars = np.asarray(model.get('pVars', []), dtype=float)
    chi2s = np.asarray(model.get('chi2s', []), dtype=float)
    alphas = np.asarray(model.get('dale_alphas', []), dtype=float)
    illegal = np.asarray(model.get('illegal_intra', []), dtype=float)
    has_extra = len(alphas) > 0 or len(illegal) > 0
    created = ax is None
    if created:
        if has_extra:
            fig, axes = plt.subplots(2, 2, figsize=(8, 6))
            axes = np.asarray(axes).ravel()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    else:
        fig = ax.figure
        axes = [ax]
    runs = np.arange(len(pVars))
    axes[0].plot(runs, pVars, marker='o', ms=3)
    axes[0].axhline(0.0, color='k', lw=0.6)
    axes[0].set_xlabel('training run')
    axes[0].set_ylabel('pVar')
    axes[0].set_title('variance explained (higher is better)')
    if len(axes) > 1:
        axes[1].plot(np.arange(len(chi2s)), chi2s, marker='o', ms=3, color='C1')
        axes[1].set_xlabel('training run')
        axes[1].set_ylabel('chi2')
        axes[1].set_title('training error (lower is better)')
    if has_extra and len(axes) > 3:
        if len(alphas):
            axes[2].plot(np.arange(len(alphas)), alphas, marker='o', ms=3, color='C2')
        axes[2].set_xlabel('training run')
        axes[2].set_ylabel('alpha')
        axes[2].set_title('intra Dale anneal (1 = hard clip)')
        axes[2].set_ylim(-0.05, 1.05)
        if len(illegal):
            axes[3].plot(np.arange(len(illegal)), illegal, marker='o', ms=3, color='C3')
        axes[3].set_xlabel('training run')
        axes[3].set_ylabel('illegal intra L2')
        axes[3].set_title('Dale violation mass (want 0)')
    if created:
        fig.tight_layout()
    return fig, axes


def plot_rate_match(model, n_trials=8, ax=None):
    """Compare target data rates to RNN rates at data times."""
    Adata, pred = model_rates_at_data(model)
    trial_len = 41
    resets = model.get('params', {}).get('resetPoints', None)
    if resets is not None and len(np.atleast_1d(resets)) > 1:
        rp = np.asarray(resets)
        dtFactor = int(model['params']['dtFactor'])
        trial_len = max(1, int(round((rp[1] - rp[0]) / float(dtFactor))))
    Tshow = min(Adata.shape[1], n_trials * trial_len)
    Ashow = Adata[:, :Tshow]
    Pshow = pred[:, :Tshow]
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 2, height_ratios=[1.1, 1.1, 1.0], hspace=0.45, wspace=0.3)
    v = np.percentile(np.abs(Ashow), 99) or 1.0
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(Ashow, aspect='auto', cmap='viridis', vmin=-v, vmax=v,
               interpolation='nearest')
    ax0.set_title('data rates (Adata)')
    ax0.set_ylabel('neuron')
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(Pshow, aspect='auto', cmap='viridis', vmin=-v, vmax=v,
               interpolation='nearest')
    ax1.set_title('RNN rates at data times')
    ax1.set_ylabel('neuron')
    for ax in (ax0, ax1):
        for t in range(trial_len, Tshow, trial_len):
            ax.axvline(t, color='w', lw=0.4, alpha=0.5)
    ax_sc = fig.add_subplot(gs[1, 0])
    step = max(1, Adata.size // 40000)
    ax_sc.scatter(Adata.ravel()[::step], pred.ravel()[::step], s=2, alpha=0.2, c='k')
    lim = max(np.percentile(np.abs(Adata), 99), np.percentile(np.abs(pred), 99), 0.1)
    ax_sc.plot([-lim, lim], [-lim, lim], 'r-', lw=0.8)
    ax_sc.set_xlabel('data')
    ax_sc.set_ylabel('model')
    ax_sc.set_title('rate match (all units/times)')
    ax_sc.set_aspect('equal', adjustable='box')
    ax_tr = fig.add_subplot(gs[1, 1])
    pops = _ordered_populations(model['populations'])
    t = np.arange(Tshow)
    for _s, _e, name, idx in pops:
        unit = int(idx[len(idx) // 2])
        ax_tr.plot(t, Ashow[unit], color=POP_COLORS.get(name, 'k'), lw=1.0,
                   label='{} u{} data'.format(name, unit))
        ax_tr.plot(t, Pshow[unit], color=POP_COLORS.get(name, 'k'), lw=1.0,
                   ls='--', alpha=0.8)
    ax_tr.set_xlabel('time bin')
    ax_tr.set_ylabel('rate')
    ax_tr.set_title('example units (solid=data, dashed=RNN)')
    ax_tr.legend(fontsize=7, ncol=2)
    ax_r2 = fig.add_subplot(gs[2, :])
    r2 = _r2_per_neuron(Adata, pred)
    colors = []
    for i in range(len(r2)):
        colors.append(POP_COLORS.get(_unit_population_name(i, model['populations']), '#888'))
    ax_r2.bar(np.arange(len(r2)), r2, color=colors, width=1.0, linewidth=0)
    ax_r2.axhline(0, color='k', lw=0.6)
    ax_r2.axhline(0.5, color='0.5', ls='--', lw=0.8)
    ax_r2.set_xlabel('readout neuron index')
    ax_r2.set_ylabel('R2')
    ax_r2.set_title('per-neuron variance explained')
    for start, _end, name, _idx in pops:
        if start > 0:
            ax_r2.axvline(start - 0.5, color='k', lw=0.6)
    return fig, (ax0, ax1, ax_sc, ax_tr, ax_r2)


def _trial_shape(n_time, trial_length=None):
    if trial_length is not None:
        trial_length = int(trial_length)
        if trial_length > 0 and n_time >= trial_length and n_time % trial_length == 0:
            return n_time // trial_length, trial_length
    for trial_len in (41, 40, 20, 11, 10, 50):
        if n_time >= trial_len and n_time % trial_len == 0:
            return n_time // trial_len, trial_len
    return 1, n_time


def _example_unit_index(idx, Adata):
    """Pick a unit whose trial-mean PSTH actually modulates (not a silent midpoint)."""
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return None
    n_trials, trial_len = _trial_shape(Adata.shape[1])
    rates = Adata[idx]
    if n_trials < 2 or rates.shape[1] != n_trials * trial_len:
        std = np.std(rates, axis=1)
        std = np.where(np.isfinite(std), std, -np.inf)
        return int(idx[int(np.argmax(std))])
    psth = rates.reshape(len(idx), n_trials, trial_len).mean(axis=1)
    mod = np.std(psth, axis=1)
    mod = np.where(np.isfinite(mod), mod, -np.inf)
    return int(idx[int(np.argmax(mod))])


def plot_population_predictions(model, n_show_trials=8, pred=None, title=None):
    """Data vs RNN rates for each pickle population (CFA_E, CFA_I, RFA_E, RFA_I).

    PSTHs are trial means. Overlay traces include whatever reset schedule
    produced `pred` (default: the stored training RNN, often teacher-forced).
    """
    Adata, pred_m = model_rates_at_data(model)
    if pred is None:
        pred = pred_m
    else:
        pred = np.asarray(pred)
        T = min(Adata.shape[1], pred.shape[1])
        Adata = Adata[:, :T]
        pred = pred[:, :T]
    n_trials, trial_len = _trial_shape(Adata.shape[1])
    dt_ms = float(model.get('dtData', 0.02)) * 1000.0
    t_ms = np.arange(trial_len) * dt_ms
    pops = _ordered_populations(model['populations'])
    r2 = _r2_per_neuron(Adata, pred)
    n_show = min(n_show_trials, n_trials)
    Tshow = n_show * trial_len

    fig = plt.figure(figsize=(11.5, 10.5))
    gs = GridSpec(len(pops), 3, hspace=0.55, wspace=0.35)
    axes = []
    for row, (_s, _e, name, idx) in enumerate(pops):
        idx = np.asarray(idx)
        color = POP_COLORS.get(name, 'k')
        A = Adata[idx]
        P = pred[idx]
        A_tr = A.reshape(len(idx), n_trials, trial_len)
        P_tr = P.reshape(len(idx), n_trials, trial_len)
        A_psth = A_tr.mean(axis=(0, 1))
        P_psth = P_tr.mean(axis=(0, 1))
        A_sem = A_tr.mean(axis=0).std(axis=0) / np.sqrt(n_trials)
        P_sem = P_tr.mean(axis=0).std(axis=0) / np.sqrt(n_trials)
        pop_r2 = float(np.mean(r2[idx]))
        corr = np.corrcoef(A.ravel(), P.ravel())[0, 1]

        ax0 = fig.add_subplot(gs[row, 0])
        ax0.fill_between(t_ms, A_psth - A_sem, A_psth + A_sem, color=color, alpha=0.2, lw=0)
        ax0.plot(t_ms, A_psth, color=color, lw=1.8, label='data')
        ax0.plot(t_ms, P_psth, color='k', lw=1.4, ls='--', label='RNN')
        ax0.set_ylabel('mean rate')
        ax0.set_title('pickle {}  n={}  medR2={:.2f}  corr={:.2f}'.format(
            name, len(idx), float(np.median(r2[idx])), corr))
        if row == 0:
            ax0.legend(fontsize=7, loc='upper right')
        if row == len(pops) - 1:
            ax0.set_xlabel('time in trial (ms)')

        ax1 = fig.add_subplot(gs[row, 1])
        t_all = np.arange(Tshow) * dt_ms
        u = _example_unit_index(idx, Adata)
        ax1.plot(t_all, Adata[u, :Tshow], color=color, lw=0.9, label='ex. unit data')
        ax1.plot(t_all, pred[u, :Tshow], color='k', lw=0.9, ls='--', label='RNN')
        for k in range(1, n_show):
            ax1.axvline(k * trial_len * dt_ms, color='0.8', lw=0.5)
        ax1.set_ylabel('rate')
        ax1.set_title('pickle {} example unit {}, {} trials'.format(name, u, n_show))
        if row == len(pops) - 1:
            ax1.set_xlabel('time (ms)')

        ax2 = fig.add_subplot(gs[row, 2])
        step = max(1, A.size // 8000)
        ax2.scatter(A.ravel()[::step], P.ravel()[::step], s=3, alpha=0.25, c=color, linewidths=0)
        lim = max(np.percentile(np.abs(A), 99), np.percentile(np.abs(P), 99), 0.05)
        ax2.plot([-lim, lim], [-lim, lim], color='0.3', lw=0.7)
        ax2.set_xlabel('data')
        ax2.set_ylabel('RNN')
        ax2.set_title('{} bins'.format(name))
        ax2.set_aspect('equal', adjustable='box')
        axes.append((ax0, ax1, ax2))
    fig.suptitle(
        title or (
            'Predictions grouped by PICKLE clusters (not learned Dale E/I). '
            'Solid=data, dashed=RNN.'),
        y=0.995, fontsize=11)
    return fig, axes


def plot_tf_vs_honest_psth(model, pred_tf, pred_honest, title=None):
    """Pickle-cluster PSTHs under 100 ms teacher-force vs trial-start resets."""
    Adata, _ = model_rates_at_data(model)
    pred_tf = np.asarray(pred_tf)
    pred_honest = np.asarray(pred_honest)
    T = min(Adata.shape[1], pred_tf.shape[1], pred_honest.shape[1])
    Adata = Adata[:, :T]
    pred_tf = pred_tf[:, :T]
    pred_honest = pred_honest[:, :T]
    n_trials, trial_len = _trial_shape(Adata.shape[1])
    dt_ms = float(model.get('dtData', 0.02)) * 1000.0
    t_ms = np.arange(trial_len) * dt_ms
    pops = _ordered_populations(model['populations'])
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    axes = axes.ravel()
    for ax, (_s, _e, name, idx) in zip(axes, pops):
        idx = np.asarray(idx)
        color = POP_COLORS.get(name, 'k')
        A_tr = Adata[idx].reshape(len(idx), n_trials, trial_len)
        T_tr = pred_tf[idx].reshape(len(idx), n_trials, trial_len)
        H_tr = pred_honest[idx].reshape(len(idx), n_trials, trial_len)
        A_psth = A_tr.mean(axis=(0, 1))
        T_psth = T_tr.mean(axis=(0, 1))
        H_psth = H_tr.mean(axis=(0, 1))
        c_tf = float(np.corrcoef(Adata[idx].ravel(), pred_tf[idx].ravel())[0, 1])
        c_h = float(np.corrcoef(Adata[idx].ravel(), pred_honest[idx].ravel())[0, 1])
        ax.plot(t_ms, A_psth, color=color, lw=1.8, label='data')
        ax.plot(t_ms, T_psth, color='k', lw=1.4, ls='--', label='TF 100 ms  corr={:.2f}'.format(c_tf))
        ax.plot(t_ms, H_psth, color='0.35', lw=1.4, ls=':', label='trial starts  corr={:.2f}'.format(c_h))
        ax.set_title('pickle {}  n={}'.format(name, len(idx)))
        ax.set_ylabel('mean rate')
        ax.legend(fontsize=7, loc='upper right')
    axes[2].set_xlabel('time in trial (ms)')
    axes[3].set_xlabel('time in trial (ms)')
    fig.suptitle(title or 'Teacher-force vs trial-start rollouts (frozen J)', fontsize=11)
    fig.tight_layout()
    return fig, axes


def plot_pseudo_opto_psth(model, opto=None, title=None, **sim_kwargs):
    """Full-trial mean rates on pseudo-opto vs control (not a 5-bin stim snippet)."""
    if opto is None:
        opto = simulate_pseudo_opto_trials(model, **sim_kwargs)
    Adata = np.asarray(opto['Adata'])
    pred_ctrl = np.asarray(opto['pred_ctrl'])
    pred_opto = np.asarray(opto['pred_opto'])
    n_trials = int(opto['n_trials'])
    trial_len = int(opto['trial_length'])
    dt_ms = float(model.get('dtData', 0.02)) * 1000.0
    t_ms = np.arange(trial_len) * dt_ms
    onset_ms = float(opto['stim_onset_s']) * 1000.0
    offset_ms = onset_ms + float(opto['dur']) * 1000.0
    target_name = opto['target_population']
    pops = _ordered_populations(model['populations'])
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    axes = axes.ravel()
    for ax, (_s, _e, name, idx) in zip(axes, pops):
        idx = np.asarray(idx)
        color = POP_COLORS.get(name, 'k')
        A_tr = Adata[idx].reshape(len(idx), n_trials, trial_len)
        C_tr = pred_ctrl[idx].reshape(len(idx), n_trials, trial_len)
        O_tr = pred_opto[idx].reshape(len(idx), n_trials, trial_len)
        A_psth = A_tr.mean(axis=(0, 1))
        C_psth = C_tr.mean(axis=(0, 1))
        O_psth = O_tr.mean(axis=(0, 1))
        O_sem = O_tr.mean(axis=0).std(axis=0) / np.sqrt(max(n_trials, 1))
        ax.axvspan(onset_ms, offset_ms, color='0.85', lw=0, zorder=0)
        ax.plot(t_ms, A_psth, color=color, lw=1.6, label='data')
        ax.plot(t_ms, C_psth, color='0.35', lw=1.3, ls=':', label='RNN control')
        ax.plot(t_ms, O_psth, color='k', lw=1.6, label='RNN opto')
        ax.fill_between(t_ms, O_psth - O_sem, O_psth + O_sem, color='k', alpha=0.12, lw=0)
        n_tgt = len(idx)
        suffix = '  TARGET' if name == target_name else ''
        ax.set_title('pickle {}  n={}{}'.format(name, n_tgt, suffix))
        ax.set_ylabel('mean rate')
        ax.legend(fontsize=7, loc='upper right')
    axes[2].set_xlabel('time in trial (ms)')
    axes[3].set_xlabel('time in trial (ms)')
    fig.suptitle(
        title or (
            'Pseudo-opto: +half-normal I stim, {:.0f} ms pulse on {} '
            '(amp={:g}), full {}-bin trials'
        ).format(
            float(opto['dur']) * 1000.0, target_name, opto['optoAmp'], trial_len),
        fontsize=11)
    fig.tight_layout()
    return fig, axes, opto


def plot_dale_scalar_diagnostics(model):
    """Gamma, E local-vs-across, and per-population R2 for a dale-scalar fit."""
    Adata, pred = model_rates_at_data(model)
    r2 = _r2_per_neuron(Adata, pred)
    J = np.asarray(model['J'])
    gamma = np.asarray(model.get('gamma', model['ei_sign'])).reshape(-1)
    gamma0 = np.asarray(model.get('gamma0', gamma)).reshape(-1)
    n1 = len(model['regions']['region1'])
    n = J.shape[0]
    e = gamma > 0
    fig = plt.figure(figsize=(11, 7.2))
    gs = GridSpec(2, 3, hspace=0.45, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(gamma0, gamma, s=12, c=np.where(e, POP_COLORS['CFA_E'], POP_COLORS['CFA_I']), alpha=0.7)
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.axhline(10.0, color='0.5', ls='--', lw=0.7)
    ax.axhline(-10.0, color='0.5', ls='--', lw=0.7)
    ax.set_xlabel('gamma init (sign of I0 intra column mean)')
    ax.set_ylabel('learned gamma')
    n_clip = int(np.sum(np.abs(gamma) >= 10.0 - 1e-6))
    ax.set_title('Dale scalar; {} units at ±10 FORCE clip'.format(n_clip))

    ax = fig.add_subplot(gs[0, 1])
    loc = np.zeros(n)
    acr = np.zeros(n)
    loc[:n1] = np.abs(J[:n1, :n1]).sum(axis=0)
    loc[n1:] = np.abs(J[n1:, n1:]).sum(axis=0)
    acr[:n1] = np.abs(J[n1:, :n1]).sum(axis=0)
    acr[n1:] = np.abs(J[:n1, n1:]).sum(axis=0)
    ax.scatter(loc[~e], acr[~e], s=10, c=POP_COLORS['CFA_I'], alpha=0.5, label='I (n={})'.format(int((~e).sum())))
    ax.scatter(loc[e], acr[e], s=18, c=POP_COLORS['CFA_E'], alpha=0.9, label='E (n={})'.format(int(e.sum())))
    ax.set_xlabel('local outgoing L1')
    ax.set_ylabel('across outgoing L1')
    ax.set_title('E can be local and across; I across = 0')
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[0, 2])
    gmax = max(10.5, float(np.max(np.abs(gamma))) + 0.5)
    bins = np.linspace(-gmax, gmax, 43)
    ax.hist(gamma[~e], bins=bins, color=POP_COLORS['CFA_I'], alpha=0.75, label='I')
    ax.hist(gamma[e], bins=bins, color=POP_COLORS['CFA_E'], alpha=0.85, label='E')
    ax.axvline(0, color='k', lw=0.6)
    ax.axvline(10.0, color='0.5', ls='--', lw=0.8)
    ax.axvline(-10.0, color='0.5', ls='--', lw=0.8)
    ax.set_xlabel('learned gamma')
    ax.set_ylabel('units')
    ax.set_title('gamma (dashed = clip ±10)')
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, :])
    pops = _ordered_populations(model['populations'])
    x = 0
    ticks = []
    labels = []
    for _s, _e, name, idx in pops:
        idx = np.asarray(idx)
        vals = np.clip(r2[idx], -1.0, 1.0)
        ax.bar(np.arange(x, x + len(vals)), vals,
               color=POP_COLORS.get(name, '#888'), width=1.0, linewidth=0)
        ticks.append(x + len(vals) / 2.0)
        labels.append('{} (med R2={:.2f}, corr={:.2f})'.format(
            name, float(np.median(r2[idx])),
            float(np.corrcoef(Adata[idx].ravel(), pred[idx].ravel())[0, 1])))
        x += len(vals)
        ax.axvline(x - 0.5, color='k', lw=0.4)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('per-neuron R2 (clipped to [-1, 1])')
    ax.set_title('Prediction R2 by pickle population (silent units explode unclipped mean R2)')
    return fig, None


def learned_region_ei_groups(model):
    """Four groups: CFA/RFA x learned Dale E/I from sign(gamma)."""
    gamma = np.asarray(model.get('gamma', model['ei_sign'])).reshape(-1)
    n1 = len(model['regions']['region1'])
    n = gamma.shape[0]
    cfa = np.arange(n) < n1
    groups = [
        ('CFA learned E', np.where(cfa & (gamma > 0))[0], POP_COLORS['CFA_E']),
        ('CFA learned I', np.where(cfa & (gamma < 0))[0], POP_COLORS['CFA_I']),
        ('RFA learned E', np.where((~cfa) & (gamma > 0))[0], POP_COLORS['RFA_E']),
        ('RFA learned I', np.where((~cfa) & (gamma < 0))[0], POP_COLORS['RFA_I']),
    ]
    return groups


def plot_learned_ei_predictions(model, n_show_trials=8):
    """Data vs RNN rates grouped by LEARNED Dale identity, not pickle tags."""
    Adata, pred = model_rates_at_data(model)
    n_trials, trial_len = _trial_shape(Adata.shape[1])
    dt_ms = float(model.get('dtData', 0.02)) * 1000.0
    t_ms = np.arange(trial_len) * dt_ms
    r2 = _r2_per_neuron(Adata, pred)
    n_show = min(n_show_trials, n_trials)
    Tshow = n_show * trial_len
    groups = learned_region_ei_groups(model)

    fig = plt.figure(figsize=(11.5, 10.5))
    gs = GridSpec(4, 3, hspace=0.55, wspace=0.35)
    for row, (name, idx, color) in enumerate(groups):
        ax0 = fig.add_subplot(gs[row, 0])
        ax1 = fig.add_subplot(gs[row, 1])
        ax2 = fig.add_subplot(gs[row, 2])
        if len(idx) == 0:
            ax0.set_title('{}  n=0'.format(name))
            continue
        A = Adata[idx]
        P = pred[idx]
        A_tr = A.reshape(len(idx), n_trials, trial_len)
        P_tr = P.reshape(len(idx), n_trials, trial_len)
        A_psth = A_tr.mean(axis=(0, 1))
        P_psth = P_tr.mean(axis=(0, 1))
        A_sem = A_tr.mean(axis=0).std(axis=0) / np.sqrt(max(n_trials, 1))
        corr = float(np.corrcoef(A.ravel(), P.ravel())[0, 1])
        med_r2 = float(np.median(r2[idx]))
        ax0.fill_between(t_ms, A_psth - A_sem, A_psth + A_sem, color=color, alpha=0.2, lw=0)
        ax0.plot(t_ms, A_psth, color=color, lw=1.8, label='data')
        ax0.plot(t_ms, P_psth, color='k', lw=1.4, ls='--', label='RNN')
        ax0.set_ylabel('mean rate')
        ax0.set_title('{}  n={}  medR2={:.2f}  corr={:.2f}'.format(
            name, len(idx), med_r2, corr))
        if row == 0:
            ax0.legend(fontsize=7, loc='upper right')
        if row == 3:
            ax0.set_xlabel('time in trial (ms)')
        t_all = np.arange(Tshow) * dt_ms
        u = _example_unit_index(idx, Adata)
        ax1.plot(t_all, Adata[u, :Tshow], color=color, lw=0.9)
        ax1.plot(t_all, pred[u, :Tshow], color='k', lw=0.9, ls='--')
        for k in range(1, n_show):
            ax1.axvline(k * trial_len * dt_ms, color='0.8', lw=0.5)
        ax1.set_ylabel('rate')
        ax1.set_title('{} example unit {}'.format(name, u))
        if row == 3:
            ax1.set_xlabel('time (ms)')
        step = max(1, A.size // 8000)
        ax2.scatter(A.ravel()[::step], P.ravel()[::step], s=3, alpha=0.25, c=color, linewidths=0)
        lim = max(np.percentile(np.abs(A), 99), np.percentile(np.abs(P), 99), 0.05)
        ax2.plot([-lim, lim], [-lim, lim], color='0.3', lw=0.7)
        ax2.set_xlabel('data')
        ax2.set_ylabel('RNN')
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_title('{} bins'.format(name))
    fig.suptitle(
        'Predictions grouped by LEARNED Dale identity (sign of gamma). '
        'Solid=data, dashed=RNN (100 ms teacher-force).',
        y=0.995, fontsize=11)
    return fig, None


def plot_learned_example_units(model, n_show_trials=8, pred=None):
    """One example unit per learned CFA/RFA E/I group. Data vs 100 ms TF only."""
    Adata, pred_m = model_rates_at_data(model)
    if pred is None:
        pred = pred_m
    else:
        pred = np.asarray(pred)
        T = min(Adata.shape[1], pred.shape[1])
        Adata = Adata[:, :T]
        pred = pred[:, :T]
    n_trials, trial_len = _trial_shape(Adata.shape[1])
    dt_ms = float(model.get('dtData', 0.02)) * 1000.0
    n_show = min(n_show_trials, n_trials)
    Tshow = n_show * trial_len
    t_all = np.arange(Tshow) * dt_ms
    groups = learned_region_ei_groups(model)
    fig, axes = plt.subplots(len(groups), 1, figsize=(11.5, 10.2), sharex=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, (name, idx, color) in zip(axes, groups):
        idx = np.asarray(idx)
        if len(idx) == 0:
            ax.set_title('{}  n=0'.format(name))
            continue
        u = _example_unit_index(idx, Adata)
        ax.plot(t_all, Adata[u, :Tshow], color=color, lw=1.2, label='data u{}'.format(u))
        ax.plot(t_all, pred[u, :Tshow], color='k', lw=1.15, ls='--', label='RNN 100 ms TF')
        for k in range(1, n_show):
            ax.axvline(k * trial_len * dt_ms, color='0.7', lw=0.8)
        ax.set_ylabel('rate')
        ax.set_title('{}  n={}  unit {}'.format(name, len(idx), u))
        ax.legend(fontsize=7, loc='upper right')
    axes[-1].set_xlabel('time (ms); vertical line = trial boundary')
    fig.suptitle(
        'Example units, one per learned population. Solid=data, dashed=RNN '
        '(100 ms teacher-force).',
        fontsize=11)
    fig.tight_layout()
    return fig, axes


def plot_ei_count_comparison(model):
    """Side-by-side pickle cluster counts vs learned Dale E/I counts."""
    gamma = np.asarray(model.get('gamma', model['ei_sign'])).reshape(-1)
    n1 = len(model['regions']['region1'])
    cfa = np.arange(gamma.size) < n1
    put = {}
    for name, idx in model['populations'].items():
        put[name] = len(np.asarray(idx))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ax = axes[0]
    cats = ['CFA', 'RFA']
    e_put = [put.get('CFA_E', 0), put.get('RFA_E', 0)]
    i_put = [put.get('CFA_I', 0), put.get('RFA_I', 0)]
    x = np.arange(len(cats))
    w = 0.35
    ax.bar(x - w / 2, e_put, w, color=POP_COLORS['CFA_E'], label='pickle E')
    ax.bar(x + w / 2, i_put, w, color=POP_COLORS['CFA_I'], label='pickle I')
    for i, v in enumerate(e_put):
        ax.text(x[i] - w / 2, v + 1, str(v), ha='center', fontsize=9)
    for i, v in enumerate(i_put):
        ax.text(x[i] + w / 2, v + 1, str(v), ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel('units')
    ax.set_title('Pickle clusters (waveform tags)\nCFA 38E/26I, RFA 109E/16I')
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(e_put + i_put) * 1.2)

    ax = axes[1]
    e_learn = [int(np.sum(cfa & (gamma > 0))), int(np.sum((~cfa) & (gamma > 0)))]
    i_learn = [int(np.sum(cfa & (gamma < 0))), int(np.sum((~cfa) & (gamma < 0)))]
    ax.bar(x - w / 2, e_learn, w, color=POP_COLORS['CFA_E'], label='learned E')
    ax.bar(x + w / 2, i_learn, w, color=POP_COLORS['CFA_I'], label='learned I')
    for i, v in enumerate(e_learn):
        ax.text(x[i] - w / 2, v + 1, str(v), ha='center', fontsize=9)
    for i, v in enumerate(i_learn):
        ax.text(x[i] + w / 2, v + 1, str(v), ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel('units')
    ax.set_title('Learned Dale identity (sign of gamma)\nCFA {}E/{}I, RFA {}E/{}I'.format(
        e_learn[0], i_learn[0], e_learn[1], i_learn[1]))
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(e_learn + i_learn) * 1.2)
    fig.suptitle('Two different partitions of the same 189 units', y=1.02, fontsize=11)
    fig.tight_layout()
    return fig, axes


def diagnose_ei_model(model, outdir=None, prefix='ei_diag', show=False):
    """Print a fit report and save J / convergence / rate-match figures."""
    metrics = print_ei_fit_report(model)
    figs = {
        'J': plot_dale_J(model)[0],
        'columns': plot_ei_source_columns(model, n_per_pop=1)[0],
        'init': plot_init_distributions(model, which='J0')[0],
        'convergence': plot_convergence(model)[0],
        'rates': plot_rate_match(model)[0],
        'pseudo_opto': plot_pseudo_opto_psth(model)[0],
    }
    saved = {}
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        for name, fig in figs.items():
            path = os.path.join(outdir, '{}_{}.png'.format(prefix, name))
            fig.savefig(path, dpi=150, bbox_inches='tight')
            saved[name] = path
            print('saved {}'.format(path))
    if not show:
        for fig in figs.values():
            plt.close(fig)
    return {'metrics': metrics, 'figures': figs, 'saved': saved}


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



