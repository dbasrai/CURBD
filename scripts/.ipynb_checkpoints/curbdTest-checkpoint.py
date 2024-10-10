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

session_path = '../../../data/co/co9/co9_12122023'
session = MPOptoClass(session_path)
session_name = session_path.split('/')[-1]

binsize=5
sigma=binsize*10
dtFactor=5
tauRNN=.03
ampInWN=.001
nRunTrain=10
num_reset=25


bounds = session.climbing_bounds[:5,:]
reg1, reg2 = session.smoother(bounds=bounds,  binsize=binsize, concat=True,
        sigma=sigma, smooth_type='causal')
#reg1, reg2 = session.binner(bounds=bounds,  binsize=binsize, concat=True)


activity = np.hstack((reg1, reg2))
z_activity = StandardScaler().fit_transform(activity)
z_activity = z_activity.T

regions={}
regions['region1'] = np.arange(0, session.num_region1)
regions['region2'] = np.arange(session.num_region1, session.num_region1 +
        session.num_region2)

seams = getSeamsFromBounds(bounds, binsize=1)
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
        nRunFree=5,
        resetPoints=resetPoints,
        plotStatus=False)

pdump(model, f'../../../picklejar/curbdTest_{session_name}_{binsize}binsize_{dtFactor}dtFactor_{tauRNN}tauRNN_{ampInWN}ampInWN.pickle')
print('finish!')
