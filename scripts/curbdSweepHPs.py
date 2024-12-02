from src.MPOptoClass import *
from src.CURBD.utils import *
import scipy
import copy
import time
import mat73 
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import random
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import permutations, compress, product
from tqdm.notebook import tqdm
import joblib
from rrpy import ReducedRankRidge
import seaborn as sns
import math

from src.CURBD import curbd

session_path = '../../../data/co/co10/co10_01242024'
session = MPOptoClass(session_path)
session_name = session_path.split('/')[-1]


binsize=5
sigma=binsize*5
dtFactor=5
tauRNN=.05
ampInWN=.001
nRunTrain=25
num_reset=100
g=1.5
g_across=1.5
P0=1


bounds = session.climbing_bounds[:30,:]
#session.threshold_FRs(threshold=.1, bounds=bounds, overwrite=True)
reg1, reg2 = session.smoother(bounds=bounds,  binsize=binsize, concat=True,
        sigma=sigma, smooth_type='causal')


activity = np.hstack((reg1, reg2))
scaler = StandardScaler()
z_activity = scaler.fit_transform(activity)
z_activity = z_activity.T

regions={}
regions['region1'] = np.arange(0, session.num_region1)
regions['region2'] = np.arange(session.num_region1, session.num_region1 +
        session.num_region2)

seams = getSeamsFromBounds(bounds, binsize=binsize/dtFactor)
temp_output=[]
start = 0
for idx in np.arange(len(seams)):
    end = seams[idx]
    temp_output.append(np.arange(start, end, num_reset))
    start=end
resetPoints = np.concatenate(temp_output)
model = curbd.trainBioMultiRegionRNN(z_activity, 
        dtData=binsize/1000,
        dtFactor=dtFactor,
        tauRNN=tauRNN,
        ampInWN=ampInWN,
        regions=regions,
        nRunTrain=nRunTrain,
        verbose=True,
        nRunFree=1,
        resetPoints=resetPoints,
        g=g,
        g_across=g_across,
        P0=P0,
        plotStatus=False)
        
model['scaler'] = scaler
pdump(model,
        f'../../../picklejar/curbd_models/curbdBio{session_name}_gx{g_across}.pickle')
print('finish!')
