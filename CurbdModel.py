from src.utils.evaluate import *
from src.utils.filters import *
from src.utils.gen_utils import *
from src.utils.torch_utils import *
from src.neural import *
from src.regression import *
from src.experiment import *
from src.analysis import *
from src.plotter import *
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import uniform
from scipy.stats import poisson
from tqdm import tqdm
from .curbd import *

class CurbdModel:

    def __init__(self, input_dict):
        for key, value in input_dict.items():
            setattr(self, key, value)
        if 'true' in input_dict:
            model = input_dict['model']
            self.sim=True
            self.climbing_bounds = np.array([[0, self.duration]])
            self.laser_bounds = self.opto_bounds #unecessary maybe
            self.only_climbing_bounds = logical2Bounds(1 - self.opto_logical)
            print('contains sim data already')
        else:
            model = input_dict
            self.sim=False
        self.model = model
        self.idx_region1 = model['regions']['region1']
        self.idx_region2 = model['regions']['region2']

    def simulate_opto(self, t, stim_frequency, ampInWN=None, optoMult=1, plot=True):
        params = self.params
        dtRNN = self.dtRNN
        tauWN = params['tauWN']
        number_units = params['number_units']
        region1 = self.regions['region1']
        region2 = self.regions['region2']
        nonLinearity = params['nonLinearity']
        tauRNN = params['tauRNN']
        Adata = self.Adata
        J = self.J

        if ampInWN is None:
            ampInWN = params['ampInWN']
        
        #generating optotimes
        starts = np.linspace(1, t, int((t-1)*stim_frequency), endpoint=False)
        stops = starts + .025 #stims last 25 ms
        opto_times = np.vstack((starts, stops)).T
            
        #we simulate a lil extra to let it stabilize
        stab_t = .1*t
        dtStab = int(stab_t / dtRNN)
        wn_t = opto_times + stab_t
        dt_wnt = wn_t / dtRNN
        dt_wnt = dt_wnt.astype(int)

        tRNN = np.arange(0, t+stab_t, dtRNN)
        wn_t_logical = bounds2Logical(dt_wnt, 
                duration=int(tRNN[-1]/dtRNN)+1)

        #ampWN = math.sqrt(tauWN/dtRNN)
        #ampWN=1 lets check if this makes a difference
        
        iWN = npr.randn(number_units, len(tRNN))
        inputWN = np.ones((number_units, len(tRNN)))
        wn_idx = np.arange(len(tRNN))[wn_t_logical.astype(bool)]#horrific
        
        for tt in range(1, len(tRNN)):
            inputWN[:, tt] = iWN[:, tt] + (inputWN[:, tt - 1] - iWN[:, tt])*np.exp(- (dtRNN / tauWN))
        
        optoInp =  np.zeros((number_units, len(tRNN))) 
        for idx, i in enumerate(wn_idx):
            indices = self.local_r2[0]
            sparse_indices = np.random.choice(indices,
                    size=int(len(indices)//1.5),replace=False)
            localz = region2[sparse_indices] 
            optoInp[localz, i] = truncnorm.rvs(a=0, b=np.inf, loc=0,
                    scale=1,size=len(localz)) 
        #randPower = np.random.rand(optoInp.shape[0], optoInp.shape[1]) * optoMult
        inputWN = ampInWN * inputWN
        optoInp = optoMult * optoInp

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

        sim = sim[:,dtStab:]
        wn_t_logical = wn_t_logical[dtStab:]
        all_inputs = inputWN[:, dtStab:] + optoInp[:, dtStab:]
        
        if plot:
            self.plot_opto_avg(sim, wn_t_logical)
                
        return sim, wn_t_logical, all_inputs

    def rates2spikes(self, rates, scale=True, poisson_mult=5):
        rates = rates.T #assuming passing in neurons x apmles
        assert rates.shape[0]>rates.shape[1], 'samples x neurons!'
            #rates should be samples x neurons
        if scale:
            scaler = self.scaler
            train_max = self.train_max
            scaled = scaler.inverse_transform(rates * train_max) * poisson_mult
            scaled[scaled<0]=0 #only positive
            scaled = poisson.rvs(size=scaled.shape, mu=scaled)
        else:
            rates_positive = rates + np.abs(np.min(rates, axis=0))
            train_max = self.train_max
            scaled = poisson.rvs(size=rates_positive.shape,
                    mu=rates_positive*train_max*poisson_mult)
        num_r1 = len(self.regions['region1'])
        num_r2 = len(self.regions['region2'])
        num_samples = scaled.shape[0]
        reg1 = scaled[:, :num_r1]
        reg2 = scaled[:, num_r1:]


        temp = np.arange(num_samples)
        reg1_train=[]
        reg2_train=[]
        for idx in np.arange(num_r1):
            reg1_train.append(temp[reg1[:, idx]>0])
        for idx in np.arange(num_r2):
            reg2_train.append(temp[reg2[:,idx]>0])

        duration = num_samples

        return reg1_train, reg2_train, duration

    def generate_ratedata(self, rates, opto_logical, all_input):
        output={}
        output['model'] = self.model

        idx_r1 = self.regions['region1']
        idx_r2 = self.regions['region2']

        rates_region1 = rates[idx_r1,:]
        rates_region2 = rates[idx_r2,:]

        output['region1'] = rates_region1.T
        output['region2'] = rates_region2.T
        output['input'] = all_input.T
        output['true'] = rates.T
        output['opto_bounds'] = logical2Bounds(opto_logical)
        output['opto_logical'] = opto_logical
        output['duration'] = rates.shape[1]

        return output

    def generate_spikedata(self, rates, opto_logical, spikes):
        assert isinstance(rates, np.ndarray), 'rates first'
        assert isinstance(spikes, tuple), 'spikes is all outputs from rates2spikes'

        output = {}
        output['model'] = self.model
        output['true'] = rates.T
        output['region1'] = spikes[0]
        output['region2'] = spikes[1]
        output['duration'] = spikes[2]
        output['opto_bounds'] = logical2Bounds(opto_logical)
        output['opto_logical'] = opto_logical

        return output

    def get_region_predics(self, activity=None, bounds=None, binsize=1,
            nlags=None):
        if activity is None:
            activity = self.true
            if bounds is not None:
                activity = self.activityInBounds(bounds, binsize, nlags)
        else:
            assert bounds is None, 'bounds not being used'
               
        tau = self.model['params']['tauRNN']
        if binsize is None:
            dtRNN = self.model['dtRNN']
        else:
            dtRNN = binsize/1000
        J = copy.deepcopy(self.model['J'])
        #np.fill_diagonal(J, 0)
        reg1 = activity[:, self.idx_region1]
        reg2 = activity[:, self.idx_region2]

        J_dstream = J[self.idx_region1,:]
        dstream = J_dstream[:, self.idx_region1]
        upstream = J_dstream[:, self.idx_region2]
        
        dstream_inp = (dstream@reg1.T) * dtRNN / tau
        upstream_inp = (upstream@reg2.T) * dtRNN / tau

        dstream_decay = ((tau-dtRNN)/tau)*reg1

        return dstream_inp.T, upstream_inp.T, dstream_decay
    
    def plot_opto_avg(self, sim, wn_t_logical):
        opto_times = logical2Bounds(wn_t_logical)
        r1_idx = self.regions['region1']
        r2_idx = self.regions['region2']
        r1_trials = []
        r2_trials = []
        sample_r1=[]
        for trial in opto_times:
            start = int(trial[0])-25
            end = start+225
            
            r1_trials.append(sim.T[start:end,r1_idx])
            r2_trials.append(sim.T[start:end,r2_idx])
            
        r1_trials = np.array(r1_trials)
        r2_trials = np.array(r2_trials)

        r1_avg, r1_sem = trial_sem(r1_trials, pre_baseline=25)
        r2_avg, r2_sem = trial_sem(r2_trials, pre_baseline=25)

        fig, ax = plot_trial_sem([r1_avg, r2_avg], [r1_sem, r2_sem], labels=['dstream', 'upstream'], pre=25)
        ax.axvline(25, linestyle='--')
        ax.set_ylabel('avg rate')
        ax.set_xlabel('ms')
        ax.set_title('25 ms negative input to upstream region')

        return fig, ax

    def plot_opto(self, sim, wn_t_logical):
        opto_times = logical2Bounds(wn_t_logical)
        starts = opto_times[:,0]
        fig, ax = plt.subplots()
        region1 = self.regions['region1']
        region2 = self.regions['region2']

        opto_sim_down = sim[region1, :] - np.expand_dims(np.average(sim[region1,:], axis=1), 1)
        opto_sim_up = sim[region2, :] -np.expand_dims(np.average(sim[region2, :], axis=1), 1)


        ax.plot(np.average(opto_sim_down, axis=0), label='downstream')
        ax.plot(np.average(opto_sim_up, axis=0), label='upstream')
        for start in starts:
            ax.axvline(start, alpha=.5, linestyle='--')
            ax.legend()

        return fig, ax


    def analyze_J(self):
        J = self.J
        J_dstream = J[self.idx_region1,:]

        dstream = J_dstream[:, self.idx_region1]
        upstream = J_dstream[:, self.idx_region2]

        dstream = np.sum(np.abs(dstream))
        upstream = np.sum(np.abs(upstream))

        return dstream, upstream

    def activityInBounds(self, bounds, binsize=1, nlags=None):
        activity = self.true
        output = bin_neural_bounds(activity, binsize, bounds)
        if nlags is not None:
            _, output = format_and_stitch_ar(output, nlags=nlags)
        else:
            output = np.concatenate(output)
        #logical = bounds2Logical(bounds, duration=self.duration).astype(bool)
        #activity = self.true[logical,:]
        return output

    def relative_J(self, activity=None, bounds=None, binsize=1, nlags=None):
        if activity is None:
            activity = self.true
            if bounds is not None:
                activity = self.activityInBounds(bounds, binsize, nlags)

        tau = self.model['params']['tauRNN']
        dtRNN = binsize/1000
        J = self.model['J']
        reg1 = activity[:, self.idx_region1]
        reg2 = activity[:, self.idx_region2]

        J_dstream = J[self.idx_region1,:]
        dstream = J_dstream[:, self.idx_region1]
        upstream = J_dstream[:, self.idx_region2]
        
        dstream_inp = (dstream@reg1.T) * dtRNN / tau
        upstream_inp = (upstream@reg2.T) * dtRNN / tau

        dstream_decay = ((tau-dtRNN)/tau)*reg1
        sum_dstream_decay = np.sum(np.abs(dstream_decay), axis=0)

        dstream_drive = np.sum(np.abs(dstream_inp), axis=1) 
        upstream_drive = np.sum(np.abs(upstream_inp), axis=1)
        return np.sum(dstream_drive), np.sum(upstream_drive), sum_dstream_decay

    def adjustLaserBounds(self, pre, post):
        laser_bounds = copy.deepcopy(self.laser_bounds)
        only_climb_bounds = copy.deepcopy(self.only_climbing_bounds)

        laser_bounds[:,0]=laser_bounds[:,0] - pre
        laser_bounds[:,1]=laser_bounds[:,1] +post

        only_climb_bounds[1:,0] = only_climb_bounds[1:,0] + post
        only_climb_bounds[:-1,1] = only_climb_bounds[:-1,1] - pre
        return laser_bounds, only_climb_bounds



