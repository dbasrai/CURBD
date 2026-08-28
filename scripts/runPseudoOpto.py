#!/usr/bin/env python
"""Run the existing I-cell opto protocol on a fitted model and save PSTHs.

Pulse shape is unchanged: +half-normal current, 25 ms. Yaml names the
stimulated region (CFA_I on co9, RFA_I on co10/co12); current is applied
to Dale-learned I cells in that region, not pickle EI IDs. PSTHs are
grouped by Dale labels.

Example:
  python scripts/runPseudoOpto.py \\
      --fit outputs/dale_horizon_balance_sparse0/co9_DALE_i0sign_reset25 \\
      --optoAmp 5
  python scripts/runPseudoOpto.py --fit path/to/model.pickle --optoAmp 5 \\
      --keep-frac 80 60 40
"""
from __future__ import print_function

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import curbd

DEFAULT_FIT = os.path.join(
    ROOT, 'outputs', 'dale_horizon_balance_sparse0', 'co9_DALE_i0sign_reset25')
DEFAULT_DATASET = os.path.join(
    ROOT, 'datasets',
    'co9_12122023_climbing_pre20ms_post180ms_binsize20_'
    'reclassEI_inact0_35_mod0_25_aligned.pkl')


def resolve_fit_pickle(path):
    path = os.path.abspath(os.path.expanduser(str(path)))
    if os.path.isdir(path):
        cand = os.path.join(path, 'model.pickle')
        if os.path.isfile(cand):
            return cand
        raise FileNotFoundError('No model.pickle in {}'.format(path))
    if not os.path.isfile(path):
        raise FileNotFoundError('Fit pickle not found: {}'.format(path))
    return path


def attach_dataset_opto(model, dataset_path):
    """Copy laser target / trial geometry from the EI dataset yaml."""
    data = curbd.load_ei_dataset(
        dataset_path, dtFactor=int(model['params'].get('dtFactor', 5)),
        smooth_sigma=0, zscore=True, max_trials=1)
    model.setdefault('populations', data['populations'])
    model['pkl_path'] = data['pkl_path']
    model['opto_target_population'] = data.get('opto_target_population')
    model['opto_corresponding_e_population'] = data.get(
        'opto_corresponding_e_population')
    model['stimulated_region'] = data.get('stimulated_region')
    model['stim_onset_s'] = data.get('stim_onset_s', 0.02)
    tlen = int(data['trial_length'])
    model['trial_length'] = tlen
    model['n_trials'] = int(model['Adata'].shape[1]) // tlen
    return model


def keep_tag(keep_frac):
    pct = int(round(100.0 * float(keep_frac)))
    return 'keep{}'.format(pct)


def parse_keep_fracs(values):
    """80 60 40 or 0.8 0.6 0.4 -> fractions in (0, 1]. Empty -> [None] (dense J)."""
    if not values:
        return [None]
    out = []
    for v in values:
        x = float(v)
        if x > 1.0:
            x = x / 100.0
        if x <= 0.0 or x > 1.0:
            raise ValueError('keep-frac must be in (0, 1] or (0, 100], got {}'.format(v))
        out.append(x)
    return out


def amp_filename(amp, keep_frac=None):
    stem = 'pseudo_opto_psth_amp{}'.format(float(amp))
    if keep_frac is not None:
        stem = '{}_{}'.format(stem, keep_tag(keep_frac))
    return '{}.png'.format(stem)


def main():
    parser = argparse.ArgumentParser(
        description='Pseudo-opto PSTH on a fitted CURBD pickle (full trial).')
    parser.add_argument('--fit', default=DEFAULT_FIT,
                        help='model.pickle or the directory that contains it')
    parser.add_argument('--dataset', default=DEFAULT_DATASET,
                        help='EI dataset used to resolve opto_target_population')
    parser.add_argument('--optoAmp', type=float, nargs='+', default=[5.0],
                        help='Pulse amplitude(s). One PNG per value, amp in the name')
    parser.add_argument('--keep-frac', type=float, nargs='*', default=None,
                        help='Keep this fraction of currently nonzero |J| '
                             '(80 60 40 or 0.8 0.6 0.4). Frozen-J prune, not a retrain. '
                             'Omit to use the dense fitted J.')
    parser.add_argument('--target', default=None,
                        help='Override target pop (default: dataset yaml, e.g. CFA_I)')
    parser.add_argument('--output-dir', default=None,
                        help='Where to save PNGs (default: same directory as the fit)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    fit_path = resolve_fit_pickle(args.fit)
    outdir = args.output_dir or os.path.dirname(fit_path)
    os.makedirs(outdir, exist_ok=True)

    model, _gt = curbd.load_ei_fit(fit_path)
    attach_dataset_opto(model, args.dataset)
    if args.target:
        model['opto_target_population'] = args.target

    target_name, target_idx = curbd.opto_target_indices(
        model, model.get('opto_target_population'))
    print('Fit {}'.format(fit_path))
    dale = curbd.dale_populations(model)
    print('  yaml target={}  Dale n_stim={}  trials={}  trial_len={}  onset={}s'.format(
        target_name, len(target_idx),
        model.get('n_trials'), model.get('trial_length'),
        model.get('stim_onset_s')))
    print('  Dale groups {}'.format(
        {k: len(v) for k, v in dale.items()}))

    import numpy as np

    keep_fracs = parse_keep_fracs(args.keep_frac)
    J0 = np.asarray(model['J'], dtype=float)
    for keep_frac in keep_fracs:
        run_model = dict(model)
        if keep_frac is None:
            print('  J dense  nnz={}'.format(int(np.sum(np.abs(J0) > 1e-12))))
            title_extra = ''
        else:
            Jsp, stats = curbd.sparsify_keep_top_connections(J0, keep_frac)
            run_model['J'] = Jsp
            print('  keep {:.0f}% of nnz: {} -> {}'.format(
                100.0 * keep_frac, stats['n_live'], stats['n_keep']))
            title_extra = ', keep top {:.0f}% nonzero J'.format(100.0 * keep_frac)
        for amp in args.optoAmp:
            title = (
                'Pseudo-opto: +half-normal I stim, 25 ms pulse on {} '
                '(amp={:g}){}, full {}-bin trials'
            ).format(
                target_name, amp, title_extra, run_model.get('trial_length'))
            fig, _axes, opto = curbd.plot_pseudo_opto_psth(
                run_model, optoAmp=amp, target_population=target_name,
                seed=args.seed, title=title)
            path = os.path.join(outdir, amp_filename(amp, keep_frac))
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print('  amp={}  saved {}'.format(opto['optoAmp'], path))


if __name__ == '__main__':
    main()
