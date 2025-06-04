from src.MPOpto import *
from src.utils.gen_utils import *
from src.utils.filters import *
from src.utils.evaluate import *
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
import signal
import sys


from src.CURBD import curbd

def handle_ctrl_c(signal, frame):
    print("Ctrl+C pressed. Executing some code...")
    pdump(model, 
            f'../../../picklejar/curbd_models/curbdBio{session_name}_gx{g_across}.pickle')

    sys.exit(0)


session_path = '../../../data/co/co10/co10_01242024'
session = MPOpto(session_path)
session_name = session_path.split('/')[-1]

binsize=5
sigma=binsize*10
dtFactor=5
tauRNN=.05
ampInWN=.001
nRunTrain=10
num_reset=100
g=1.5
g_across= 1.5
g_loc = (-.1,-.1)
sparse_percent=60
P0=1.0


bounds = session.climbing_bounds[50:,:]
session.threshold_FRs(threshold=.75, bounds=bounds, overwrite=True)
session.subsampleNeurons(percent_region1=1, percent_region2=.6)
reg1, reg2 = session.smoother(bounds=bounds,  binsize=binsize, concat=True,
        sigma=sigma, smooth_type='causal')


activity = np.hstack((reg1, reg2))
#activity = np.hstack((reg2, reg1))
scaler = StandardScaler()
z_activity = scaler.fit_transform(activity)
z_activity = z_activity.T

regions={}
regions['region1'] = np.arange(0, session.num_region1)
regions['region2'] = np.arange(session.num_region1, session.num_region1 +
        session.num_region2)
#regions['region1'] = np.arange(0, session.num_region2)
#regions['region2'] = np.arange(session.num_region2, session.num_region1 +
#        session.num_region2)

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
        sparse_percent=sparse_percent,
        g_loc=g_loc,
        plotStatus=False)

        
model['scaler'] = scaler
pdump(model,
        f'../../../picklejar/curbd_models2/curbdBioco10_bal.pickle')
print('finish!')
