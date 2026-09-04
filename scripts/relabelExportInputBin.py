#!/usr/bin/env python
"""Copy a Dale label export and move the pseudo-opto input 1-hot.

Does not re-simulate the RNN or redraw Poisson spikes. J r currents, spike
counts, and climbing files are copied as-is. Only the exported unit input
event (and stim_bin metadata) is moved, default 1 -> 0.

Example:
  python scripts/relabelExportInputBin.py \\
    --src outputs/dale_label_export/co9_12122023_climbing \\
    --dst outputs/dale_label_export_inputBin0/co9_12122023_climbing
"""
from __future__ import print_function

import argparse
import os
import pickle
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import exportDaleCurrents as cur_export
import exportDaleLabeledSpikes as spike_export


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def relabel_currents_file(src, dst, from_bin, to_bin):
    blob = load_pickle(src)
    if blob.get('condition') != 'pseudo_opto':
        shutil.copy2(src, dst)
        js_src = src[:-7] + '.json' if src.endswith('.pickle') else None
        if js_src and os.path.isfile(js_src):
            shutil.copy2(js_src, dst[:-7] + '.json')
        return blob.get('stim_bin')
    out = cur_export.relabel_current_input_event(blob, from_bin, to_bin)
    cur_export.write_export(out, os.path.dirname(dst), copy_dir=None)
    return out.get('stim_bin')


def relabel_spikes_file(src, dst, from_bin, to_bin):
    blob = load_pickle(src)
    out = spike_export.relabel_spike_input_event(blob, from_bin, to_bin)
    spike_export.write_pickle(dst, out)
    return out.get('stim_bin')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src',
        default=os.path.join(
            ROOT, 'outputs', 'dale_label_export', 'co9_12122023_climbing'))
    parser.add_argument(
        '--dst',
        default=os.path.join(
            ROOT, 'outputs', 'dale_label_export_inputBin0',
            'co9_12122023_climbing'))
    parser.add_argument('--from-bin', type=int, default=1)
    parser.add_argument('--to-bin', type=int, default=0)
    args = parser.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    if not os.path.isdir(src):
        raise ValueError('Missing source export {}'.format(src))
    if os.path.abspath(src) == dst:
        raise ValueError('Refusing to overwrite the source folder in place')
    os.makedirs(dst, exist_ok=True)

    names = sorted(
        n for n in os.listdir(src)
        if n.endswith('.pickle') or n.endswith('.json'))
    print('Relabel {} -> {}  input bin {} -> {}'.format(
        src, dst, args.from_bin, args.to_bin))
    for name in names:
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        if name.endswith('_curbdCurrents_daleEI.pickle'):
            stim = relabel_currents_file(s, d, args.from_bin, args.to_bin)
            print('  currents {}  stim_bin={}'.format(name, stim))
        elif name.endswith('_curbdCurrents_daleEI.json'):
            continue
        elif name.endswith('_spikeCounts_daleEI.pickle'):
            stim = relabel_spikes_file(s, d, args.from_bin, args.to_bin)
            print('  spikes {}  stim_bin={}'.format(name, stim))
        else:
            shutil.copy2(s, d)
            print('  copy {}'.format(name))


if __name__ == '__main__':
    main()
