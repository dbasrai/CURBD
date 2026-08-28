#!/usr/bin/env python
"""Teacher-force horizon + per-region E-count Dale-scalar on climbing EI pickles.

Identity is assigned by ranking I0 intra column means inside each region, so
CFA E count is a chosen floor rather than whatever sign(mean) infers (often ~5).
Compare to the I0 / Bio ceiling on the same reset schedule, not to pVar 0.5.
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

CO9 = os.path.join(
    ROOT, 'datasets',
    'co9_12122023_climbing_pre20ms_post180ms_binsize20_'
    'reclassEI_inact0_35_mod0_25_aligned.pkl')
CO10 = os.path.join(
    ROOT, 'datasets',
    'co10_01252024_climbing_pre20ms_post180ms_binsize20_'
    'reclassEI_inact0_35_mod0_25_aligned.pkl')
CO12 = os.path.join(
    ROOT, 'datasets',
    'co12_02132024_climbing_pre20ms_post180ms_binsize20_'
    'reclassEI_inact0_35_mod0_25_aligned.pkl')

I0_DIR = os.path.join(ROOT, 'outputs', 'constraint_ladder_co9')

BASE = dict(
    g=1.5, g_across=1.5, g_loc=(-0.1, -0.1),
    tauRNN=0.2, ampInWN=0.001, P0=1.0,
    sparse_percent=60, dtFactor=5, nRunTrain=40, nRunFree=1,
)


def region_fit(model):
    A, P = curbd.model_rates_at_data(model)
    n1 = len(model['regions']['region1'])
    g = np.asarray(model.get('gamma', model['ei_sign'])).reshape(-1)
    out = {}
    for name, idx in (
            ('CFA', np.arange(n1)),
            ('RFA', np.arange(n1, A.shape[0]))):
        a, p = A[idx], P[idx]
        r2 = curbd._r2_per_neuron(a, p)
        std = float(a.std())
        dist = np.linalg.norm(a - p)
        pvar = 1.0 - (dist / (np.sqrt(a.size) * (std + 1e-12))) ** 2
        corr = float(np.corrcoef(a.ravel(), p.ravel())[0, 1])
        out[name] = dict(
            n=int(len(idx)), pVar=float(pvar), corr=corr,
            medR2=float(np.median(r2)),
            nE=int(np.sum(g[idx] > 0)), nI=int(np.sum(g[idx] < 0)),
        )
    return out


def train_i0(data, reset_every, nRunTrain, seed=0, sparse_percent=60):
    npr.seed(seed)
    np.random.seed(seed)
    resetPoints = curbd.make_reset_points(
        data['n_trials'], data['trial_length'], 5, reset_every=reset_every)
    return curbd.trainBioConstrainedRNN(
        data['z_activity'], dtData=data['dtData'], dtFactor=5,
        tauRNN=0.2, ampInWN=0.001, regions=data['regions'],
        nRunTrain=nRunTrain, nRunFree=1, resetPoints=resetPoints,
        g=1.5, g_across=1.5, g_loc=(-0.1, -0.1), P0=1.0,
        sparse_percent=sparse_percent, ei_sign=data['ei_sign'],
        populations=data['populations'], zero_i_across=True,
        dale_scalar=False, plotStatus=False, verbose=True,
    )


def train_dale(data, J_init, raw, nRunTrain, seed=0, sparse_percent=60):
    npr.seed(seed)
    np.random.seed(seed)
    resetPoints = curbd.make_reset_points(
        data['n_trials'], data['trial_length'], 5,
        reset_every=raw.get('reset_every'))
    n_E = raw.get('n_E')
    if n_E == 'pickle':
        n_E = (len(data['populations']['CFA_E']),
               len(data['populations']['RFA_E']))
    return curbd.trainBioConstrainedRNN(
        data['z_activity'], dtData=data['dtData'], dtFactor=5,
        tauRNN=0.2, ampInWN=0.001, regions=data['regions'],
        nRunTrain=nRunTrain, nRunFree=1, resetPoints=resetPoints,
        g=1.5, g_across=1.5, g_loc=(-0.1, -0.1), P0=1.0,
        sparse_percent=sparse_percent, ei_sign=data['ei_sign'],
        populations=data['populations'], zero_i_across=True,
        dale_scalar=True, J_init=J_init,
        gamma_init=raw['gamma_init'],
        e_frac=raw.get('e_frac'), n_E=n_E,
        min_e_frac=raw.get('min_e_frac'),
        gamma_gain=raw.get('gamma_gain', 1.0),
        gamma_l2=raw.get('gamma_l2', 0.0),
        epoch_flip=bool(raw.get('epoch_flip', False)),
        flip_margin=raw.get('flip_margin', 0.2),
        plotStatus=False, verbose=True,
    )


def summarize(model, name, elapsed, n_reset, extra):
    metrics = curbd.summarize_ei_fit(model)
    n1 = len(model['regions']['region1'])
    g = np.asarray(model['gamma']).reshape(-1)
    g0 = np.asarray(model['gamma0']).reshape(-1)
    blocks = curbd.constraint_block_metrics(model['J'], np.sign(g), n1)
    regs = region_fit(model)
    pv = np.asarray(metrics['pVars'], dtype=float)
    row = dict(
        name=name,
        nRunTrain=int(model['params']['nRunTrain']),
        n_reset=int(n_reset),
        pvar0=float(pv[0]),
        pvar_peak=float(np.max(pv)),
        pvar_final=float(pv[-1]),
        corr=metrics['corr'],
        sat_model=metrics['sat_model'],
        n_E=int(np.sum(g > 0)),
        n_I=int(np.sum(g < 0)),
        n_flip=int(np.sum(np.sign(g) != np.sign(g0))),
        pickle_agree=float(np.mean(np.sign(g) == np.sign(model['ei_sign_init']))),
        n_gclip=int(np.sum(np.abs(g) >= 9.99)),
        g_abs_mean=float(np.mean(np.abs(g))),
        g_abs_max=float(np.max(np.abs(g))),
        j_col_l1_E=float(np.abs(model['J'])[:, g > 0].sum(0).mean()) if np.any(g > 0) else float('nan'),
        j_col_l1_I=float(np.abs(model['J'])[:, g < 0].sum(0).mean()) if np.any(g < 0) else float('nan'),
        mixed_col=blocks['mixed_col_frac'],
        i_across_l1=blocks['i_across_l1'],
        CFA_nE=regs['CFA']['nE'], CFA_nI=regs['CFA']['nI'],
        CFA_pVar=regs['CFA']['pVar'], CFA_corr=regs['CFA']['corr'],
        CFA_medR2=regs['CFA']['medR2'],
        RFA_nE=regs['RFA']['nE'], RFA_nI=regs['RFA']['nI'],
        RFA_pVar=regs['RFA']['pVar'], RFA_corr=regs['RFA']['corr'],
        RFA_medR2=regs['RFA']['medR2'],
        elapsed_s=elapsed,
    )
    row.update(extra)
    tf = curbd.evaluate_teacher_force(model)
    for r in tf:
        row['pvar_' + r['schedule']] = r['pVar']
        row['corr_' + r['schedule']] = r['corr']
    return row, tf


def export_generator_labels(model, run_dir):
    """CSV of Dale generator labels vs pickle tags."""
    import csv as csvlib
    g = np.asarray(model['gamma']).reshape(-1)
    g0 = np.asarray(model.get('gamma0', g)).reshape(-1)
    put = np.asarray(model['ei_sign_init']).reshape(-1)
    n1 = len(model['regions']['region1'])
    path = os.path.join(run_dir, 'synthetic_labels.csv')
    with open(path, 'w', newline='') as f:
        w = csvlib.writer(f)
        w.writerow(['unit', 'region', 'dale_label', 'dale_gamma',
                    'dale_label_init', 'pickle_label', 'flipped'])
        for i in range(g.size):
            region = 'CFA' if i < n1 else 'RFA'
            dale = 'E' if g[i] > 0 else 'I'
            dale0 = 'E' if g0[i] > 0 else 'I'
            pickle_lab = 'E' if put[i] > 0 else 'I'
            w.writerow([i, region, dale, float(g[i]), dale0, pickle_lab,
                        int(np.sign(g[i]) != np.sign(g0[i]))])
    return path


def save_run(outdir, name, model, row, tf):
    run_dir = os.path.join(outdir, name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'model.pickle'), 'wb') as f:
        pickle.dump({'model': model, 'metrics': row}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    slim = [{k: v for k, v in r.items() if k not in ('pred', 'RNN')} for r in tf]
    with open(os.path.join(run_dir, 'tf_eval.json'), 'w') as f:
        json.dump(slim, f, indent=2)
    import matplotlib.pyplot as plt
    curbd.plot_convergence(model)[0].savefig(
        os.path.join(run_dir, 'convergence.png'), dpi=120, bbox_inches='tight')
    curbd.plot_learned_ei_predictions(model)[0].savefig(
        os.path.join(run_dir, 'predictions_LEARNED_Dale_EI.png'),
        dpi=120, bbox_inches='tight')
    curbd.plot_learned_example_units(model)[0].savefig(
        os.path.join(run_dir, 'example_units.png'),
        dpi=120, bbox_inches='tight')
    curbd.plot_dale_scalar_diagnostics(model)[0].savefig(
        os.path.join(run_dir, 'dale_scalar_diagnostics.png'),
        dpi=120, bbox_inches='tight')
    export_generator_labels(model, run_dir)
    plt.close('all')


def configs_for(dataset_tag, have_co9_i0=False):
    """Dale configs. I0 is trained separately per dataset/schedule."""
    rows = [
        dict(name='{}_DALE_i0sign_reset25'.format(dataset_tag),
             reset_every=25, gamma_init='i0_sign',
             why='I0 column-sign identities; 100 ms TF.'),
        dict(name='{}_DALE_i0sign_reset50'.format(dataset_tag),
             reset_every=50, gamma_init='i0_sign',
             why='I0 column-sign identities; 200 ms TF.'),
        dict(name='{}_DALE_i0sign_reset100'.format(dataset_tag),
             reset_every=100, gamma_init='i0_sign',
             why='I0 column-sign identities; 400 ms TF.'),
        dict(name='{}_DALE_i0rank_pickleN_reset25'.format(dataset_tag),
             reset_every=25, gamma_init='i0_rank', n_E='pickle',
             why='Most E-like I0 columns, pickle E counts per region.'),
        dict(name='{}_DALE_i0rank_half_reset25'.format(dataset_tag),
             reset_every=25, gamma_init='i0_rank', e_frac=0.5,
             why='Most E-like I0 columns, 50% E in CFA and in RFA.'),
        dict(name='{}_DALE_i0rank_half_reset50'.format(dataset_tag),
             reset_every=50, gamma_init='i0_rank', e_frac=0.5,
             why='50% E per region; 200 ms TF.'),
        dict(name='{}_DALE_i0rank_half_flip_reset25'.format(dataset_tag),
             reset_every=25, gamma_init='i0_rank', e_frac=0.5,
             epoch_flip=True, min_e_frac=0.4, flip_margin=0.2,
             why='50% E init; epoch dJ flips with 40% E floor per region.'),
        dict(name='{}_DALE_i0rank_80_20_flip_reset25'.format(dataset_tag),
             reset_every=25, gamma_init='i0_rank', e_frac=0.8,
             epoch_flip=True, min_e_frac=0.7, flip_margin=0.2,
             why='80/20 E/I init; epoch dJ flips with 70% E floor per region.'),
        dict(name='{}_DALE_i0rank_half_flip_trialstarts'.format(dataset_tag),
             reset_every=None, gamma_init='i0_rank', e_frac=0.5,
             epoch_flip=True, min_e_frac=0.4, flip_margin=0.2,
             why='50% E + flips, trial-start training (generator protocol).'),
        dict(name='{}_DALE_i0rank_80_20_flip_trialstarts'.format(dataset_tag),
             reset_every=None, gamma_init='i0_rank', e_frac=0.8,
             epoch_flip=True, min_e_frac=0.7, flip_margin=0.2,
             why='80/20 + flips, trial-start training (generator protocol).'),
    ]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=CO9)
    parser.add_argument('--output-dir', default=os.path.join(
        ROOT, 'outputs', 'dale_horizon_balance'))
    parser.add_argument('--max-trials', type=int, default=40)
    parser.add_argument('--nRunTrain', type=int, default=40)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--only', default=None)
    parser.add_argument('--tag', default=None,
                        help='Name prefix (default: dataset stem co9/co10/...)')
    parser.add_argument('--sparse-percent', type=float, default=60,
                        help='Zero this percent of weakest E-across columns each '
                             'FORCE epoch (20 keeps ~80%% of E-across). '
                             '0 = dense E-across. Also used for the I0 warm-start.')
    args = parser.parse_args()

    data = curbd.load_ei_dataset(
        args.dataset, dtFactor=5, smooth_sigma=1.5, zscore=True,
        max_trials=args.max_trials)
    tag = args.tag
    if tag is None:
        tag = data['name'].split('_')[0]
    pops = {k: len(np.asarray(v)) for k, v in data['populations'].items()}
    print('data {}  N={}  CFA {} ({}E/{}I)  RFA {} ({}E/{}I)  trials={}'.format(
        data['name'], data['z_activity'].shape[0],
        pops['CFA_E'] + pops['CFA_I'], pops['CFA_E'], pops['CFA_I'],
        pops['RFA_E'] + pops['RFA_I'], pops['RFA_E'], pops['RFA_I'],
        data['n_trials']))

    only = None
    if args.only:
        only = set(x.strip() for x in args.only.split(','))

    os.makedirs(args.output_dir, exist_ok=True)
    cfgs = configs_for(tag)
    i0_local = os.path.join(
        args.output_dir, '{}_I0_reset25'.format(tag), 'model.pickle')
    i0_path_reuse = None
    if os.path.isfile(i0_local):
        i0_path_reuse = i0_local
    elif (float(args.sparse_percent) == 60
            and tag == 'co9' and 'co9_12122023' in data['name']):
        i0_path_reuse = os.path.join(I0_DIR, 'I0_reset25', 'model.pickle')
    if i0_path_reuse and os.path.isfile(i0_path_reuse):
        with open(i0_path_reuse, 'rb') as f:
            J_i0 = pickle.load(f)['model']['J']
        print('reuse I0 J_init from', i0_path_reuse)
    else:
        print('\n==== train I0_reset25 for', tag,
              'sparse_percent={}'.format(args.sparse_percent), '====')
        t0 = time.time()
        i0 = train_i0(data, 25, args.nRunTrain, seed=args.seed,
                      sparse_percent=args.sparse_percent)
        J_i0 = i0['J']
        i0_dir = os.path.join(args.output_dir, '{}_I0_reset25'.format(tag))
        os.makedirs(i0_dir, exist_ok=True)
        with open(os.path.join(i0_dir, 'model.pickle'), 'wb') as f:
            pickle.dump({'model': i0, 'metrics': {
                'pvar_final': float(np.asarray(i0['pVars'])[-1]),
                'elapsed_s': time.time() - t0,
                'sparse_percent': float(args.sparse_percent),
            }}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('I0 pVar', i0['pVars'][-1], 'in', round(time.time() - t0), 's')

    rows = []
    for raw in cfgs:
        if only and raw['name'] not in only:
            continue
        re = raw.get('reset_every')
        J_init = J_i0
        print('\n====', raw['name'], '====')
        print(raw['why'])
        t0 = time.time()
        model = train_dale(data, J_init, raw, args.nRunTrain, seed=args.seed,
                           sparse_percent=args.sparse_percent)
        model['scaler'] = data['scaler']
        model['dataset_name'] = data['name']
        model['hp_name'] = raw['name']
        elapsed = time.time() - t0
        n_reset = len(curbd.make_reset_points(
            data['n_trials'], data['trial_length'], 5, reset_every=re))
        extra = dict(
            dataset=data['name'], tag=tag,
            gamma_init=raw['gamma_init'],
            reset_every=re,
            e_frac=raw.get('e_frac'),
            min_e_frac=raw.get('min_e_frac'),
            epoch_flip=bool(raw.get('epoch_flip', False)),
            gamma_l2=raw.get('gamma_l2', 0.0),
            sparse_percent=float(args.sparse_percent),
            n_E_target=str(raw.get('n_E')),
        )
        row, tf = summarize(model, raw['name'], elapsed, n_reset, extra)
        print('RESULT {name}: pVar {pvar_final:.3f} corr={corr:.3f} '
              'CFA {CFA_nE}E/{CFA_nI}I pVar={CFA_pVar:.3f}  '
              'RFA {RFA_nE}E/{RFA_nI}I pVar={RFA_pVar:.3f}  '
              'flips={n_flip} |g|_mean={g_abs_mean:.2f} n_clip={n_gclip} '
              'honest={pvar_trialstarts:.3f}'.format(**row))
        save_run(args.output_dir, raw['name'], model, row, tf)
        rows.append(row)

    if rows:
        log_name = '{}_partial_sweep_log.csv'.format(tag) if only else '{}_sweep_log.csv'.format(tag)
        log_path = os.path.join(args.output_dir, log_name)
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(log_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print('\nWrote', log_path)


if __name__ == '__main__':
    main()
