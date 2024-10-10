from src.utils.gen_utils import *
from src.utils.filters import *
from src.experiment import *
from src.wiener_filter import *
from src.modeller import *
from src.MPRecordClass import *
from src.ReachingClass import *
from src.MPOptoCtrlClass import *
from src.analysis import *
from src.plotter import *
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

binsize=5
sigma=binsize*20
dtFactor=5
tauRNN=.03
ampInWN=.001
nRunTrain=500


reg1, reg2 = session.smoother(bounds=session.climbing_bounds, 
        sigma=sigma, binsize=binsize, concat=True, smooth_type='both')

activity = np.hstack((reg1, reg2))
z_activity = StandardScaler().fit_transform(activity)
z_activity = z_activity.T

regions={}
regions['region1'] = np.arange(0, session.num_region1)
regions['region2'] = np.arange(session.num_region1, session.num_region1 +
        session.num_region2)

model = curbd.trainMultiRegionRNN(z_activity, 
        dtData=binsize/1000,
        dtFactor=dtFactor,
        tauRNN=tauRNN,
        ampInWN=ampInWN,
        regions=regions,
        nRunTrain=nRunTrain,
        verbose=True,
        nRunFree=5,
        plotStatus=False)

pdump(model, f'../../../picklejar/curbdco9_{binsize}binsize_{dtFactor}dtFactor_{tauRNN}tauRNN_{ampInWN}ampInWN.pickle')
print('finish!')
