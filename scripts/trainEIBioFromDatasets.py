#!/usr/bin/env python
"""Train a Dale-constrained two-region CURBD RNN on CFA/RFA x E/I datasets.

Inter-region weights are sparse excitatory from E cells only.
Intra-region Dale can be off, annealed (default), or hard from init.

Fitted J and exported E/I currents are ground truth for this RNN, not
recovered synapses of the recorded animal.
"""
from __future__ import print_function

import argparse
import os
import pickle
import sys

import numpy as np
import numpy.random as npr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import curbd

DEFAULT_DATASET = os.path.join(
    ROOT, 'datasets',
    'co9_12122023_climbing_pre20ms_post180ms_binsize20_'
    'reclassEI_inact0_35_mod0_25_aligned.pkl')


def parse_g_loc(value):
    parts = [p.strip() for p in str(value).split(',')]
    if len(parts) == 1:
        x = float(parts[0])
        return (x, x)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('g_loc must be g or g1,g2')
    return (float(parts[0]), float(parts[1]))


def parse_ramp(value):
    parts = [p.strip() for p in str(value).split(',')]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('intra_dale_ramp must be start,end (e.g. 0.3,0.8)')
    return (float(parts[0]), float(parts[1]))


def main():
    parser = argparse.ArgumentParser(
        description='Fit Dale-constrained CURBD on /datasets CFA/RFA E/I pickles.')
    parser.add_argument('--dataset', default=DEFAULT_DATASET,
                        help='Dataset stem, .pkl, or .yaml (default: co10 climbing reclassEI)')
    parser.add_argument('--output-dir', default=os.path.join(ROOT, 'outputs'),
                        help='Directory for pickled model + ground-truth export')
    parser.add_argument('--nRunTrain', type=int, default=10)
    parser.add_argument('--nRunFree', type=int, default=1)
    parser.add_argument('--dtFactor', type=int, default=5)
    parser.add_argument('--tauRNN', type=float, default=0.2)
    parser.add_argument('--ampInWN', type=float, default=0.001)
    parser.add_argument('--g', type=float, default=1.5)
    parser.add_argument('--g_across', type=float, default=1.5)
    parser.add_argument('--g_loc', type=parse_g_loc, default=(-0.1, -0.1))
    parser.add_argument('--sparse_percent', type=float, default=0)
    parser.add_argument('--P0', type=float, default=1.0)
    parser.add_argument('--adata-scale', default='z3',
                        choices=['max', 'z3', 'p99', 'clip'],
                        help='How to map z-scored rates into tanh range (z3 recommended)')
    parser.add_argument('--target-radius', type=float, default=1.2,
                        help='Rescale init J to this spectral radius (e.g. 1.2)')
    parser.add_argument('--max-radius', type=float, default=1.2,
                        help='Rescale J during training if spectral radius exceeds this')
    parser.add_argument('--max-radius-after-frac', type=float, default=None,
                        help='Only apply max-radius after this fraction of nRunTrain (e.g. 0.66)')
    parser.add_argument('--init-intra', default='signed', choices=['signed', 'dale'],
                        help='signed: Bio Gaussian intra (then anneal). dale: folded E/I from init')
    parser.add_argument('--intra-dale', default='anneal', choices=['off', 'anneal', 'hard'],
                        help='off: no intra Dale. anneal: leak violations to 0. hard: clip every step')
    parser.add_argument('--intra-dale-ramp', type=parse_ramp, default=(0.3, 0.8),
                        help='Fraction of nRunTrain to start,end the intra Dale ramp')
    parser.add_argument('--project-intra-when', default='update',
                        choices=['update', 'reset', 'epoch'],
                        help='When to apply intra Dale leak (update=every RLS)')
    parser.add_argument('--max-trials', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--reset-every', type=int, default=None,
                        help='Extra teacher-force resets every N RNN steps (Bio used 100)')
    parser.add_argument('--reset-state', default='rate', choices=['rate', 'current'],
                        help='rate: H=Adata (legacy). current: H=arctanh(Adata)')
    parser.add_argument('--align-error', default='prev', choices=['prev', 'current'],
                        help='FORCE error vs previous data bin (legacy) or current bin')
    parser.add_argument('--smooth_sigma', type=float, default=1.5,
                        help='Causal Gaussian sigma in bins (0 disables)')
    parser.add_argument('--t-sim', type=float, default=None,
                        help='Autonomous simulation duration in seconds (default: training length)')
    parser.add_argument('--poisson', action='store_true',
                        help='Also export Poisson spikes from simulated rates')
    parser.add_argument('--plotStatus', action='store_true')
    parser.add_argument('--diagnose', action='store_true',
                        help='After fitting, save J / convergence / rate-match figures')
    args = parser.parse_args()

    npr.seed(args.seed)
    np.random.seed(args.seed)

    data = curbd.load_ei_dataset(
        args.dataset,
        dtFactor=args.dtFactor,
        smooth_sigma=args.smooth_sigma,
        zscore=True,
        max_trials=args.max_trials,
        reset_every=args.reset_every)
    print('Loaded {}  N={} T={} trials={} trial_len={} dtData={}s'.format(
        data['name'], data['z_activity'].shape[0], data['z_activity'].shape[1],
        data['n_trials'], data['trial_length'], data['dtData']))
    for name, idx in data['populations'].items():
        print('  {}: {} units'.format(name, len(idx)))

    model = curbd.trainEIBioMultiRegionRNN(
        data['z_activity'],
        dtData=data['dtData'],
        dtFactor=args.dtFactor,
        tauRNN=args.tauRNN,
        ampInWN=args.ampInWN,
        regions=data['regions'],
        populations=data['populations'],
        ei_sign=data['ei_sign'],
        nRunTrain=args.nRunTrain,
        verbose=True,
        nRunFree=args.nRunFree,
        resetPoints=data['resetPoints'],
        g=args.g,
        g_across=args.g_across,
        P0=args.P0,
        sparse_percent=args.sparse_percent,
        g_loc=args.g_loc,
        plotStatus=args.plotStatus,
        adata_scale=args.adata_scale,
        target_radius=args.target_radius,
        max_radius=args.max_radius,
        max_radius_after_frac=args.max_radius_after_frac,
        reset_state=args.reset_state,
        align_error=args.align_error,
        init_intra=args.init_intra,
        intra_dale=args.intra_dale,
        intra_dale_ramp=args.intra_dale_ramp,
        project_intra_when=args.project_intra_when)
    model['scaler'] = data['scaler']
    model['dataset_name'] = data['name']
    model['pkl_path'] = data['pkl_path']

    dale = curbd.check_dale_constraints(model)
    print('Dale check: intra E min={:.4g}  intra I max={:.4g}  '
          'inter E min={:.4g}  I->across absmax={:.4g}  ok={}'.format(
              dale['intra_e_min'], dale['intra_i_max'],
              dale['inter_e_min'], dale['inter_i_absmax'], dale['ok']['all']))
    print('final pVar={} chi2={}'.format(model['pVars'][-1], model['chi2s'][-1]))

    gt = curbd.export_ei_groundtruth(
        model, t=args.t_sim, poisson_spikes=args.poisson)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'curbdEI_{}.pickle'.format(data['name']))
    with open(out_path, 'wb') as f:
        pickle.dump({'model': model, 'ground_truth': gt}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Wrote {}'.format(out_path))

    if args.diagnose:
        curbd.diagnose_ei_model(
            model, outdir=args.output_dir,
            prefix='ei_diag_{}'.format(data['name']), show=False)


if __name__ == '__main__':
    main()
