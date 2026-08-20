#!/usr/bin/env python
"""Validate a fitted Dale-constrained CURBD model.

Prints a convergence / rate-match report and saves:
  - J heatmap with CFA/RFA x E/I blocks (columns = sources, rows = readouts)
  - example E and I outgoing columns vs readout neuron index
  - pVar / chi2 training curves
  - data vs RNN rate rasters, scatter, traces, and per-neuron R2

pVar should rise toward 1 (usable fit is typically > 0.5). A largely negative
pVar with model rates stuck at +/-1 means the RNN did not track the data.
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
import curbd

DEFAULT_FIT = os.path.join(
    ROOT, 'outputs',
    'curbdEI_co10_01242024_climbing_pre20ms_post180ms_binsize20_'
    'reclassEI_inact0_35_mod0_25_aligned.pickle')


def main():
    parser = argparse.ArgumentParser(description='Diagnose a fitted EI CURBD pickle.')
    parser.add_argument('--fit', default=DEFAULT_FIT, help='Pickle from trainEIBioFromDatasets.py')
    parser.add_argument('--output-dir', default=None,
                        help='Where to save PNGs (default: same directory as the pickle)')
    parser.add_argument('--prefix', default='ei_diag')
    args = parser.parse_args()

    model, _gt = curbd.load_ei_fit(args.fit)
    outdir = args.output_dir or os.path.dirname(os.path.abspath(args.fit)) or ROOT
    print('Loaded {}'.format(args.fit))
    print('N={}  T_data={}  T_RNN={}  nRunTrain={}'.format(
        model['J'].shape[0], model['Adata'].shape[1], model['RNN'].shape[1],
        model['params'].get('nRunTrain')))
    for name, idx in model['populations'].items():
        print('  {}: units {}-{} (n={})'.format(
            name, int(idx[0]), int(idx[-1]), len(idx)))
    curbd.diagnose_ei_model(model, outdir=outdir, prefix=args.prefix, show=False)


if __name__ == '__main__':
    main()
