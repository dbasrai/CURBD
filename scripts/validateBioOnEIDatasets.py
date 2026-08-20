#!/usr/bin/env python
"""Validate trainBioMultiRegionRNN on the 20 ms CFA/RFA E/I pickles.

This is the original Bio CURBD: signed intra-region J, excitatory (clipped)
inter-region from *all* source units, optional weak-column sparsify.
It does not know E vs I and does not zero I→across.

If this cannot reach pVar >= 0.5, Dale is not the first bottleneck.
"""
from __future__ import print_function

import argparse
import csv
import os
import pickle
import sys
import time

import numpy as np
import numpy.random as npr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use('Agg')
import curbd

DEFAULT_DATASET = os.path.join(
    ROOT, 'datasets',
    'co9_12122023_climbing_pre20ms_post180ms_binsize20_'
    'reclassEI_inact0_35_mod0_25_aligned.pkl')

# Literal HPs from scripts/curbdSweepHPs.py, then 20 ms timing variants.
BASE = dict(
    g=1.5,
    g_across=1.5,
    g_loc=(-0.1, -0.1),
    tauRNN=0.05,
    ampInWN=0.001,
    P0=1.0,
    sparse_percent=60,
    dtFactor=5,
    reset_every=100,
    nRunTrain=15,
    nRunFree=1,
)

CONFIGS = [
    dict(
        name='BIO_literal_5msHPs',
        why=(
            'Closest replay of curbdSweepHPs.py on this pickle: z-score, '
            'Bio does /max internally, tauRNN=0.05, num_reset=100 RNN steps, '
            'sparse 60%. 20 ms bins make dtRNN=4 ms and resets every 400 ms.'
        ),
    ),
    dict(
        name='BIO_tau02',
        tauRNN=0.2,
        why='Keep Bio resets/sparsity; set tau/dtData=10 like 5 ms Bio (tau=0.05 / 5 ms).',
    ),
    dict(
        name='BIO_tau02_reset50',
        tauRNN=0.2,
        reset_every=50,
        why=(
            '20 ms x dtFactor=5 x reset_every=50 is 200 ms, matching 10 ms x '
            'dtFactor=5 x num_reset=100.'
        ),
    ),
    dict(
        name='BIO_tau02_reset25',
        tauRNN=0.2,
        reset_every=25,
        why=(
            '20 ms x dtFactor=5 x reset_every=25 is 100 ms, matching the 5 ms '
            'Bio script (dtRNN=1 ms, num_reset=100).'
        ),
    ),
    dict(
        name='BIO_tau02_trial_resets',
        tauRNN=0.2,
        reset_every=None,
        why='Trial starts only. Isolates whether Bio needs mid-trial teacher-force on this data.',
    ),
    dict(
        name='BIO_tau02_dense_cross',
        tauRNN=0.2,
        reset_every=100,
        sparse_percent=0,
        why='Excitatory cross from all units, no sparsify. Tests if 60% prune is required.',
    ),
]


def merge_cfg(raw):
    cfg = dict(BASE)
    for k, v in raw.items():
        if k not in ('name', 'why'):
            cfg[k] = v
    cfg['name'] = raw['name']
    cfg['why'] = raw['why']
    return cfg


def run_one(data, cfg, outdir, seed=0):
    npr.seed(seed)
    np.random.seed(seed)
    dtFactor = int(cfg['dtFactor'])
    resetPoints = curbd.make_reset_points(
        data['n_trials'], data['trial_length'], dtFactor,
        reset_every=cfg.get('reset_every'))
    print('\n' + '=' * 72)
    print('CONFIG', cfg['name'])
    print('WHY:', cfg['why'])
    print('HPs:', {k: cfg[k] for k in (
        'tauRNN', 'dtFactor', 'reset_every', 'sparse_percent', 'P0',
        'g', 'g_across', 'nRunTrain')})
    print('resetPoints: n={}  trial_rnn={}  example={}'.format(
        len(resetPoints), data['trial_length'] * dtFactor, resetPoints[:8]))
    t0 = time.time()
    model = curbd.trainBioMultiRegionRNN(
        data['z_activity'],
        dtData=data['dtData'],
        dtFactor=dtFactor,
        tauRNN=cfg['tauRNN'],
        ampInWN=cfg['ampInWN'],
        regions=data['regions'],
        nRunTrain=cfg['nRunTrain'],
        nRunFree=cfg['nRunFree'],
        resetPoints=resetPoints,
        g=cfg['g'],
        g_across=cfg['g_across'],
        P0=cfg['P0'],
        sparse_percent=cfg['sparse_percent'],
        g_loc=cfg['g_loc'],
        plotStatus=False,
        verbose=True,
    )
    model['scaler'] = data['scaler']
    model['dataset_name'] = data['name']
    model['hp_name'] = cfg['name']
    model['populations'] = data['populations']
    model['ei_sign'] = data['ei_sign']
    elapsed = time.time() - t0
    metrics = curbd.summarize_ei_fit(model)
    rho = curbd._spectral_radius(model['J'])
    J = np.asarray(model['J'])
    n1 = len(data['regions']['region1'])
    cross_neg_frac = float(np.mean(np.concatenate([
        J[:n1, n1:].ravel() < 0, J[n1:, :n1].ravel() < 0])))
    row = dict(
        name=cfg['name'],
        tauRNN=cfg['tauRNN'],
        dtFactor=dtFactor,
        reset_every=cfg.get('reset_every'),
        sparse_percent=cfg['sparse_percent'],
        nRunTrain=cfg['nRunTrain'],
        n_reset=int(len(resetPoints)),
        train_max=float(model.get('train_max', np.nan)),
        Adata_std=metrics['Adata_std'],
        pred_std=metrics['pred_std'],
        pvar0=float(metrics['pVars'][0]) if len(metrics['pVars']) else np.nan,
        pvar_final=metrics['pvar_final'],
        chi2_final=metrics['chi2_final'],
        corr=metrics['corr'],
        sat_model=metrics['sat_model'],
        rho_final=rho,
        cross_neg_frac=cross_neg_frac,
        status=metrics['status'],
        elapsed_s=elapsed,
    )
    print('RESULT {name}: status={status} pVar {pvar0:.3f}->{pvar_final:.3f} '
          'corr={corr:.3f} sat={sat_model:.1%} rho={rho_final:.2f} '
          'Astd={Adata_std:.4f} train_max={train_max:.3g} {elapsed_s:.0f}s'.format(**row))
    run_dir = os.path.join(outdir, cfg['name'])
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'model.pickle'), 'wb') as f:
        pickle.dump({'model': model, 'metrics': row}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    curbd.plot_convergence(model)[0].savefig(
        os.path.join(run_dir, 'convergence.png'), dpi=120, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close('all')
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--output-dir', default=os.path.join(
        ROOT, 'outputs', 'bio_validate'))
    parser.add_argument('--max-trials', type=int, default=40)
    parser.add_argument('--nRunTrain', type=int, default=15)
    parser.add_argument('--only', default=None)
    args = parser.parse_args()

    only = None
    if args.only:
        only = set(x.strip() for x in args.only.split(','))

    data = curbd.load_ei_dataset(
        args.dataset, dtFactor=5, smooth_sigma=1.5, zscore=True,
        max_trials=args.max_trials)
    print('Bio validation data {}  N={} T={} trials={} dtData={}s'.format(
        data['name'], data['z_activity'].shape[0], data['z_activity'].shape[1],
        data['n_trials'], data['dtData']))
    print('regions: CFA {}  RFA {}'.format(
        len(data['regions']['region1']), len(data['regions']['region2'])))
    print('z_activity max={:.3g} std={:.3g} (Bio will divide by max)'.format(
        np.max(np.abs(data['z_activity'])), data['z_activity'].std()))

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for raw in CONFIGS:
        if only and raw['name'] not in only:
            continue
        cfg = merge_cfg(raw)
        cfg['nRunTrain'] = args.nRunTrain
        rows.append(run_one(data, cfg, args.output_dir))

    log_path = os.path.join(args.output_dir, 'sweep_log.csv')
    if rows:
        with open(log_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print('\nWrote', log_path)
        print('\nBIO VALIDATION (sorted by pVar):')
        for r in sorted(rows, key=lambda x: x['pvar_final'], reverse=True):
            print('  {name:32s}  pVar={pvar_final:8.3f}  corr={corr:6.3f}  '
                  'sat={sat_model:6.1%}  status={status}'.format(**r))


if __name__ == '__main__':
    main()
