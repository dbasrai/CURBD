#!/usr/bin/env python
"""Dale via learned per-source scalar: J[:,j] = gamma[j] * softplus(W[:,j]).

Each column is purely E (gamma>0) or I (gamma<0), never mixed. gamma is
unconstrained and may flip relative to putative pickle labels. I-across is 0
for learned I cells. Compare to the I0 ceiling, not to pVar 0.5.
"""
from __future__ import print_function

import argparse
import csv
import json
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

I0_DIR = os.path.join(ROOT, 'outputs', 'constraint_ladder_co9')

BASE = dict(
    g=1.5,
    g_across=1.5,
    g_loc=(-0.1, -0.1),
    tauRNN=0.2,
    ampInWN=0.001,
    P0=1.0,
    sparse_percent=60,
    dtFactor=5,
    nRunTrain=40,
    nRunFree=1,
    zero_i_across=True,
    dale_scalar=True,
)

CONFIGS = [
    dict(
        name='DALE_pickle_reset25',
        reset_every=25,
        gamma_init='pickle',
        warm_start=None,
        why='Putative pickle E/I as gamma init; Bio J projected onto that cone.',
    ),
    dict(
        name='DALE_i0sign_reset25',
        reset_every=25,
        gamma_init='i0_sign',
        warm_start='I0_reset25',
        why='Warm-start I0 J; gamma init = sign of I0 intra column means.',
    ),
    dict(
        name='DALE_i0sign_reset25_bothproj',
        reset_every=25,
        gamma_init='i0_sign',
        warm_start='I0_reset25',
        why='I0-sign Dale without E-across column sparsify: E can be local AND long-range.',
    ),
    dict(
        name='DALE_pickle_trialstarts',
        reset_every=None,
        gamma_init='pickle',
        warm_start=None,
        why='Pickle gamma init; trial-start resets only.',
    ),
    dict(
        name='DALE_i0sign_trialstarts',
        reset_every=None,
        gamma_init='i0_sign',
        warm_start='I0_trialstarts',
        why='Warm-start I0_trialstarts J; gamma from I0 column signs.',
    ),
    dict(
        name='DALE_pickle_reset25_long',
        reset_every=25,
        gamma_init='pickle',
        warm_start=None,
        nRunTrain=120,
        why='Same pickle cone as the failed 40-run fit, trained 3x longer. No extra flips.',
    ),
    dict(
        name='DALE_pickle_reset25_longflip',
        reset_every=25,
        gamma_init='pickle',
        warm_start=None,
        nRunTrain=120,
        gamma_gain=8.0,
        epoch_flip=True,
        flip_every=None,
        flip_margin=0.2,
        why='Pickle init, 120 runs, LS gamma_gain=8 plus one cone reassignment per run from FORCE dJ.',
    ),
    dict(
        name='DALE_pickle_trialstarts_longflip',
        reset_every=None,
        gamma_init='pickle',
        warm_start=None,
        nRunTrain=120,
        gamma_gain=8.0,
        epoch_flip=True,
        flip_every=None,
        flip_margin=0.2,
        why='Pickle init with epoch-end dJ cone flips, trial-start resets only (honest TF).',
    ),
]


def load_warm_J(name):
    path = os.path.join(I0_DIR, name, 'model.pickle')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            'Need I0 warm-start pickle at {}. Run ablateConstraints.py first.'.format(path))
    with open(path, 'rb') as f:
        return pickle.load(f)['model']['J']


def save_eval_plots(model, run_dir, n_trials, trial_length, tag=''):
    import matplotlib.pyplot as plt
    os.makedirs(run_dir, exist_ok=True)
    prefix = (tag + '_') if tag else ''
    tf_rows = curbd.evaluate_teacher_force(
        model, n_trials=n_trials, trial_length=trial_length)
    tf_slim = [{k: v for k, v in r.items() if k not in ('pred', 'RNN')}
               for r in tf_rows]
    with open(os.path.join(run_dir, prefix + 'tf_eval.json'), 'w') as f:
        json.dump(tf_slim, f, indent=2)
    print('  TF eval:')
    for r in tf_slim:
        print('    {schedule:16s}  n_reset={n_reset:4d}  pVar={pVar:8.3f}  '
              'corr={corr:6.3f}  sat={sat:.1%}'.format(**r))
    by_name = {r['schedule']: r for r in tf_rows}
    curbd.plot_population_predictions(
        model,
        title='PICKLE clusters on this model (training RNN; usually teacher-forced)',
    )[0].savefig(os.path.join(run_dir, prefix + 'predictions_PICKLE_clusters.png'),
                 dpi=120, bbox_inches='tight')
    curbd.plot_learned_ei_predictions(model)[0].savefig(
        os.path.join(run_dir, prefix + 'predictions_LEARNED_Dale_EI.png'),
        dpi=120, bbox_inches='tight')
    if 'reset25_100ms' in by_name and 'trialstarts' in by_name:
        curbd.plot_tf_vs_honest_psth(
            model, by_name['reset25_100ms']['pred'], by_name['trialstarts']['pred'],
            title='Frozen J: 100 ms teacher-force vs trial-start resets',
        )[0].savefig(os.path.join(run_dir, prefix + 'tf_vs_trialstarts_psth.png'),
                     dpi=120, bbox_inches='tight')
    curbd.plot_convergence(model)[0].savefig(
        os.path.join(run_dir, prefix + 'convergence.png'), dpi=120, bbox_inches='tight')
    plt.close('all')
    return tf_slim


def run_one(data, raw, outdir, nRunTrain, seed=0):
    cfg = dict(BASE)
    cfg.update({k: v for k, v in raw.items() if k not in ('why',)})
    if 'nRunTrain' not in raw:
        cfg['nRunTrain'] = nRunTrain
    cfg['why'] = raw['why']
    cfg.setdefault('gamma_gain', 1.0)
    cfg.setdefault('epoch_flip', False)
    cfg.setdefault('flip_every', None)
    cfg.setdefault('flip_margin', 0.05)

    npr.seed(seed)
    np.random.seed(seed)
    dtFactor = int(cfg['dtFactor'])
    resetPoints = curbd.make_reset_points(
        data['n_trials'], data['trial_length'], dtFactor,
        reset_every=cfg.get('reset_every'))
    J_init = None
    if cfg.get('warm_start'):
        J_init = load_warm_J(cfg['warm_start'])

    print('\n' + '=' * 72)
    print('CONFIG', cfg['name'])
    print('WHY:', cfg['why'])
    print('gamma_init={}  warm_start={}  n_reset={}  nRunTrain={}  '
          'gamma_gain={}  epoch_flip={}  flip_every={}'.format(
              cfg['gamma_init'], cfg.get('warm_start'), len(resetPoints),
              cfg['nRunTrain'], cfg['gamma_gain'], cfg['epoch_flip'],
              cfg.get('flip_every')))
    t0 = time.time()
    model = curbd.trainBioConstrainedRNN(
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
        g_loc=cfg['g_loc'],
        P0=cfg['P0'],
        sparse_percent=cfg['sparse_percent'],
        ei_sign=data['ei_sign'],
        populations=data['populations'],
        zero_i_across=True,
        dale_scalar=True,
        J_init=J_init,
        gamma_init=cfg['gamma_init'],
        gamma_gain=cfg['gamma_gain'],
        epoch_flip=cfg['epoch_flip'],
        flip_every=cfg.get('flip_every'),
        flip_margin=cfg['flip_margin'],
        plotStatus=False,
        verbose=True,
    )
    model['scaler'] = data['scaler']
    model['dataset_name'] = data['name']
    model['hp_name'] = cfg['name']
    elapsed = time.time() - t0

    metrics = curbd.summarize_ei_fit(model)
    n1 = len(data['regions']['region1'])
    learned = np.asarray(model['ei_sign']).reshape(-1)
    blocks = curbd.constraint_block_metrics(model['J'], learned, n1)
    putative = np.asarray(data['ei_sign']).reshape(-1)
    gamma = np.asarray(model['gamma']).reshape(-1)
    gamma0 = np.asarray(model['gamma0']).reshape(-1)
    n_flip = int(np.sum(np.sign(gamma) != np.sign(gamma0)))
    pickle_agree = float(np.mean(np.sign(gamma) == np.sign(putative)))
    rho = curbd._spectral_radius(model['J'])
    pvars = np.asarray(metrics['pVars'], dtype=float)
    row = dict(
        name=cfg['name'],
        gamma_init=cfg['gamma_init'],
        warm_start=cfg.get('warm_start') or '',
        reset_every=cfg.get('reset_every'),
        nRunTrain=cfg['nRunTrain'],
        n_reset=int(len(resetPoints)),
        gamma_gain=cfg['gamma_gain'],
        epoch_flip=cfg['epoch_flip'],
        pvar0=float(pvars[0]) if len(pvars) else np.nan,
        pvar_peak=float(np.max(pvars)) if len(pvars) else np.nan,
        pvar_peak_run=int(np.argmax(pvars)) if len(pvars) else -1,
        pvar_final=metrics['pvar_final'],
        corr=metrics['corr'],
        sat_model=metrics['sat_model'],
        rho_final=rho,
        i_across_l1=blocks['i_across_l1'],
        intra_sign_viol_frac=blocks['intra_sign_viol_frac'],
        mixed_col_frac=blocks['mixed_col_frac'],
        n_E=int(np.sum(gamma > 0)),
        n_I=int(np.sum(gamma < 0)),
        n_flip=n_flip,
        pickle_agree=pickle_agree,
        Adata_std=metrics['Adata_std'],
        pred_std=metrics['pred_std'],
        elapsed_s=elapsed,
    )
    print('RESULT {name}: pVar {pvar0:.3f}->{pvar_final:.3f} peak={pvar_peak:.3f}@run{pvar_peak_run} '
          'corr={corr:.3f} sat={sat_model:.1%} mixedCol={mixed_col_frac:.3f} '
          'nE={n_E} nI={n_I} flips={n_flip} pickle_agree={pickle_agree:.3f} '
          '{elapsed_s:.0f}s'.format(**row))
    run_dir = os.path.join(outdir, cfg['name'])
    os.makedirs(run_dir, exist_ok=True)
    tf_slim = save_eval_plots(model, run_dir, data['n_trials'], data['trial_length'])
    for r in tf_slim:
        row['pvar_' + r['schedule']] = r['pVar']
        row['corr_' + r['schedule']] = r['corr']
    with open(os.path.join(run_dir, 'model.pickle'), 'wb') as f:
        pickle.dump({'model': model, 'metrics': row}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    return row


def eval_existing(outdir, data, names=None):
    import matplotlib.pyplot as plt
    rows = []
    pred_dir = os.path.join(outdir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    for name in names or []:
        path = os.path.join(outdir, name, 'model.pickle')
        if not os.path.isfile(path):
            print('SKIP missing', path)
            continue
        with open(path, 'rb') as f:
            blob = pickle.load(f)
        model = blob['model']
        print('\n=== eval existing', name, '===')
        tf_slim = save_eval_plots(
            model, os.path.join(outdir, name),
            data['n_trials'], data['trial_length'], tag='posthoc')
        curbd.plot_population_predictions(
            model,
            title='{}  PICKLE clusters (training RNN)'.format(name),
        )[0].savefig(os.path.join(pred_dir, name + '_PICKLE_clusters.png'),
                     dpi=120, bbox_inches='tight')
        plt.close('all')
        row = dict(name=name)
        for r in tf_slim:
            row['pvar_' + r['schedule']] = r['pVar']
            row['corr_' + r['schedule']] = r['corr']
            row['sat_' + r['schedule']] = r['sat']
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--output-dir', default=os.path.join(
        ROOT, 'outputs', 'dale_scalar_co9'))
    parser.add_argument('--max-trials', type=int, default=40)
    parser.add_argument('--nRunTrain', type=int, default=40)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--only', default=None)
    parser.add_argument('--eval-existing', default=None,
                        help='Comma-separated run names to TF-eval without retraining')
    args = parser.parse_args()

    only = None
    if args.only:
        only = set(x.strip() for x in args.only.split(','))

    data = curbd.load_ei_dataset(
        args.dataset, dtFactor=5, smooth_sigma=1.5, zscore=True,
        max_trials=args.max_trials)
    print('Dale-scalar data {}  N={} T={} trials={}'.format(
        data['name'], data['z_activity'].shape[0], data['z_activity'].shape[1],
        data['n_trials']))
    print('putative E={} I={}'.format(
        int(np.sum(data['ei_sign'] > 0)), int(np.sum(data['ei_sign'] < 0))))

    os.makedirs(args.output_dir, exist_ok=True)
    if args.eval_existing:
        names = [x.strip() for x in args.eval_existing.split(',') if x.strip()]
        rows = eval_existing(args.output_dir, data, names)
        log_path = os.path.join(args.output_dir, 'tf_eval_existing.csv')
        if rows:
            with open(log_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            print('\nWrote', log_path)
        return

    rows = []
    for raw in CONFIGS:
        if only and raw['name'] not in only:
            continue
        rows.append(run_one(data, raw, args.output_dir, args.nRunTrain,
                            seed=args.seed))

    log_name = 'sweep_log_long.csv' if only else 'sweep_log.csv'
    log_path = os.path.join(args.output_dir, log_name)
    if rows:
        with open(log_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print('\nWrote', log_path)
        print('\nDALE SCALAR vs I0 ceiling (pVar ~0.36 reset25 / ~0.17 trialstarts):')
        for r in rows:
            print('  {name:36s}  pVar={pvar_final:8.3f}  peak={pvar_peak:8.3f}  '
                  'corr={corr:6.3f}  flips={n_flip:3d}  agree={pickle_agree:.3f}'.format(**r))


if __name__ == '__main__':
    main()
