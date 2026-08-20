#!/usr/bin/env python
"""Sequential constraint ladder on co9: UNC -> BIO -> I0 -> REPARAM.

Each step adds one constraint. Shared data, scale, and HPs. Do not mix knobs.

UNC  trainMultiRegionRNN: signed dense J (already /max internally, like Bio)
BIO  trainBioMultiRegionRNN: signed intra, excitatory sparse cross from all units
I0   Bio + I->across identically 0 (E-only long-range; FORCE still on J)
REPARAM  I0 + intra Dale via softplus W (FORCE on W)

Compare pVar to Bio on the same reset schedule, not to 0.5.
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
)

SCHEDULES = [
    dict(
        suffix='reset25',
        reset_every=25,
        why='100 ms teacher-force; Bio ceiling on this pickle.',
    ),
    dict(
        suffix='trialstarts',
        reset_every=None,
        why='Trial starts only; honest fit without mid-trial teacher-force.',
    ),
]

STEPS = [
    dict(
        step='UNC',
        trainer='multi',
        why='Unconstrained CURBD: signed Gaussian J, dense, no Bio cross.',
    ),
    dict(
        step='BIO',
        trainer='bio',
        why='Bio: signed intra, excitatory sparse cross from all units.',
    ),
    dict(
        step='I0',
        trainer='constrained',
        zero_i_across=True,
        intra_reparam=False,
        why='Bio plus I->across=0. Intra still signed; FORCE still on J.',
    ),
    dict(
        step='REPARAM',
        trainer='constrained',
        zero_i_across=True,
        intra_reparam=True,
        why='I0 plus intra Dale J_E=softplus(W), J_I=-softplus(W).',
    ),
]


def configs():
    out = []
    for sched in SCHEDULES:
        for step in STEPS:
            cfg = dict(BASE)
            cfg.update(step)
            cfg['reset_every'] = sched['reset_every']
            cfg['name'] = '{}_{}'.format(step['step'], sched['suffix'])
            cfg['why'] = '{} {}'.format(step['why'], sched['why'])
            out.append(cfg)
    return out


def attach_labels(model, data, cfg):
    model['scaler'] = data['scaler']
    model['dataset_name'] = data['name']
    model['hp_name'] = cfg['name']
    model['populations'] = data['populations']
    model['ei_sign'] = data['ei_sign']
    return model


def train_step(data, cfg, resetPoints):
    trainer = cfg['trainer']
    common = dict(
        dtData=data['dtData'],
        dtFactor=int(cfg['dtFactor']),
        tauRNN=cfg['tauRNN'],
        ampInWN=cfg['ampInWN'],
        regions=data['regions'],
        nRunTrain=cfg['nRunTrain'],
        nRunFree=cfg['nRunFree'],
        resetPoints=resetPoints,
        g=cfg['g'],
        P0=cfg['P0'],
        plotStatus=False,
        verbose=True,
    )
    z = data['z_activity']
    if trainer == 'multi':
        return curbd.trainMultiRegionRNN(z, **common)
    if trainer == 'bio':
        return curbd.trainBioMultiRegionRNN(
            z, g_across=cfg['g_across'], g_loc=cfg['g_loc'],
            sparse_percent=cfg['sparse_percent'], **common)
    if trainer == 'constrained':
        return curbd.trainBioConstrainedRNN(
            z, g_across=cfg['g_across'], g_loc=cfg['g_loc'],
            sparse_percent=cfg['sparse_percent'],
            ei_sign=data['ei_sign'],
            populations=data['populations'],
            zero_i_across=cfg['zero_i_across'],
            intra_reparam=cfg['intra_reparam'],
            **common)
    raise ValueError('Unknown trainer {}'.format(trainer))


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
        'step', 'trainer', 'tauRNN', 'dtFactor', 'reset_every',
        'sparse_percent', 'P0', 'g', 'g_across', 'nRunTrain')})
    print('resetPoints: n={}  trial_rnn={}  example={}'.format(
        len(resetPoints), data['trial_length'] * dtFactor, resetPoints[:8]))
    t0 = time.time()
    model = train_step(data, cfg, resetPoints)
    attach_labels(model, data, cfg)
    elapsed = time.time() - t0
    metrics = curbd.summarize_ei_fit(model)
    n1 = len(data['regions']['region1'])
    blocks = curbd.constraint_block_metrics(
        model['J'], data['ei_sign'], n1)
    rho = curbd._spectral_radius(model['J'])
    row = dict(
        name=cfg['name'],
        step=cfg['step'],
        trainer=cfg['trainer'],
        tauRNN=cfg['tauRNN'],
        dtFactor=dtFactor,
        reset_every=cfg.get('reset_every'),
        sparse_percent=cfg['sparse_percent'] if cfg['trainer'] != 'multi' else 0,
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
        i_across_l1=blocks['i_across_l1'],
        intra_sign_viol_frac=blocks['intra_sign_viol_frac'],
        cross_neg_frac=blocks['cross_neg_frac'],
        status=metrics['status'],
        elapsed_s=elapsed,
    )
    print('RESULT {name}: pVar {pvar0:.3f}->{pvar_final:.3f} corr={corr:.3f} '
          'sat={sat_model:.1%} IacrossL1={i_across_l1:.3g} '
          'intraViol={intra_sign_viol_frac:.3f} crossNeg={cross_neg_frac:.3f} '
          '{elapsed_s:.0f}s'.format(**row))
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


def print_ladder(rows):
    print('\nCONSTRAINT LADDER (by schedule, sequential steps):')
    header = (
        '{name:24s}  {pvar:>8s}  {corr:>6s}  {sat:>6s}  '
        '{iL1:>10s}  {viol:>8s}  {cneg:>6s}'.format(
            name='name', pvar='pVar', corr='corr', sat='sat',
            iL1='IacrossL1', viol='intraViol', cneg='xNeg'))
    print(header)
    for r in rows:
        print('  {name:24s}  {pvar_final:8.3f}  {corr:6.3f}  {sat_model:6.1%}  '
              '{i_across_l1:10.3g}  {intra_sign_viol_frac:8.3f}  '
              '{cross_neg_frac:6.3f}'.format(**r))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--output-dir', default=os.path.join(
        ROOT, 'outputs', 'constraint_ladder_co9'))
    parser.add_argument('--max-trials', type=int, default=40)
    parser.add_argument('--nRunTrain', type=int, default=40)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--only', default=None,
                        help='Comma-separated config names, e.g. I0_reset25')
    args = parser.parse_args()

    only = None
    if args.only:
        only = set(x.strip() for x in args.only.split(','))

    data = curbd.load_ei_dataset(
        args.dataset, dtFactor=5, smooth_sigma=1.5, zscore=True,
        max_trials=args.max_trials)
    print('Ladder data {}  N={} T={} trials={} dtData={}s'.format(
        data['name'], data['z_activity'].shape[0], data['z_activity'].shape[1],
        data['n_trials'], data['dtData']))
    print('regions: CFA {}  RFA {}'.format(
        len(data['regions']['region1']), len(data['regions']['region2'])))
    print('z_activity max={:.3g} std={:.3g} '
          '(UNC and Bio both divide by max internally)'.format(
              np.max(np.abs(data['z_activity'])), data['z_activity'].std()))

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for cfg in configs():
        if only and cfg['name'] not in only:
            continue
        cfg['nRunTrain'] = args.nRunTrain
        rows.append(run_one(data, cfg, args.output_dir, seed=args.seed))

    log_path = os.path.join(args.output_dir, 'sweep_log.csv')
    if rows:
        with open(log_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print('\nWrote', log_path)
        print_ladder(rows)


if __name__ == '__main__':
    main()
