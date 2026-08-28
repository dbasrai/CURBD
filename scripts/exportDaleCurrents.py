#!/usr/bin/env python
"""Export Dale pop-to-pop CURBD currents for use in another repo.

Fits are climbing-only. Pseudo-opto is a post-hoc amp-5 I-cell pulse on the
fitted J. Climbing and perturb are written as separate files; RMS aggregation
is computed inside each file from that condition's currents only.

  {animal}_{date}_climbingOnly_{init}_curbdCurrents_daleEI.pickle
  {animal}_{date}_pseudoOptoOnly_{init}_amp{amp}_curbdCurrents_daleEI.pickle

Each pickle
-----------
condition : 'climbing' or 'pseudo_opto'

influence_rms[source][target] : float
    Mean over target units of RMS_t(J[target, source] @ r_source),
    aggregated from this condition only.

currents_over_time[source][target] : (T,) float32
    Unit- and trial-mean J r vs time.

currents_per_trial[source][target] : (n_trials, T) float32
    Same current, mean over target units, trials kept.

Sources: CFA_E, CFA_I, RFA_E, RFA_I, input.
Targets: CFA_E, CFA_I, RFA_E, RFA_I.
"""
from __future__ import print_function

import argparse
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import curbd

CO9 = os.path.join(
    ROOT, 'datasets',
    'co9_12122023_climbing_pre20ms_post180ms_binsize20_'
    'reclassEI_inact0_35_mod0_25_aligned.pkl')

DEFAULT_FITS = (
    ('i0init', os.path.join(
        ROOT, 'outputs', 'dale_horizon_balance_sparse0',
        'co9_DALE_i0sign_reset25', 'model.pickle')),
    ('50_50', os.path.join(
        ROOT, 'outputs', 'dale_horizon_balance_sparse80',
        'co9_DALE_i0rank_half_flip_reset25', 'model.pickle')),
)

POPS = list(curbd.EI_POP_ORDER)
SOURCES = POPS + ['input']


def session_tag_from_dataset(pkl_path):
    stem = os.path.splitext(os.path.basename(pkl_path))[0]
    parts = stem.split('_')
    if len(parts) >= 3:
        return '_'.join(parts[:3])
    return stem


def animal_date_from_session(session):
    parts = str(session).split('_')
    if len(parts) >= 2:
        return '_'.join(parts[:2])
    return str(session)


def load_model(path):
    with open(path, 'rb') as f:
        blob = pickle.load(f)
    if isinstance(blob, dict) and 'model' in blob:
        return blob['model']
    if isinstance(blob, dict) and 'J' in blob:
        return blob
    raise ValueError('Unrecognized model pickle: {}'.format(path))


def attach_opto(model, dataset_path):
    data = curbd.load_ei_dataset(
        dataset_path, dtFactor=int(model['params'].get('dtFactor', 5)),
        smooth_sigma=0, zscore=True, max_trials=1)
    model = dict(model)
    model.setdefault('populations', data['populations'])
    model['opto_target_population'] = data.get('opto_target_population')
    model['stimulated_region'] = data.get('stimulated_region')
    model['stim_onset_s'] = data.get('stim_onset_s', 0.02)
    tlen = int(data['trial_length'])
    model['trial_length'] = tlen
    model['n_trials'] = int(np.asarray(model['Adata']).shape[1]) // tlen
    return model


def sample_drive(wn, extra, n_data, dtFactor, shift):
    n = wn.shape[0]
    extra = np.zeros_like(wn) if extra is None else extra
    out = np.zeros((n, n_data), dtype=np.float32)
    for i in range(n_data):
        tt = i * dtFactor
        out[:, i] = (
            curbd._input_at(wn, tt, dtFactor, shift)
            + curbd._input_at(extra, tt, dtFactor, shift)
        ).ravel()
    return out


def pop_current(J, dale, rates, src, trg):
    tidx = np.asarray(dale[trg], dtype=int)
    sidx = np.asarray(dale[src], dtype=int)
    if tidx.size == 0:
        return np.zeros((1, rates.shape[1]), dtype=np.float32)
    if sidx.size == 0:
        return np.zeros((tidx.size, rates.shape[1]), dtype=np.float32)
    return np.asarray(J[np.ix_(tidx, sidx)].dot(rates[sidx]), dtype=np.float32)


def reshape_trials(I, n_tr, tlen):
    return I.reshape(I.shape[0], n_tr, tlen)


def unit_rms(I):
    return float(np.mean(np.sqrt(np.mean(np.asarray(I, dtype=float) ** 2, axis=1))))


def nest_sources_targets(factory):
    out = {}
    for src in SOURCES:
        out[src] = {}
        for trg in POPS:
            out[src][trg] = factory()
    return out


def stack_rms(nested):
    arr = np.zeros((len(SOURCES), len(POPS)), dtype=np.float32)
    for j, src in enumerate(SOURCES):
        for k, trg in enumerate(POPS):
            arr[j, k] = nested[src][trg]
    return arr


def stack_time(nested, T):
    arr = np.zeros((len(SOURCES), len(POPS), T), dtype=np.float32)
    for j, src in enumerate(SOURCES):
        for k, trg in enumerate(POPS):
            arr[j, k] = nested[src][trg]
    return arr


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def currents_for_tape(J, dale, rates, drive, n_tr, tlen):
    """Per-condition currents, then RMS aggregated from those currents only."""
    rms = nest_sources_targets(lambda: 0.0)
    over_time = nest_sources_targets(lambda: None)
    per_trial = nest_sources_targets(lambda: None)
    for trg in POPS:
        tidx = np.asarray(dale[trg], dtype=int)
        for src in POPS:
            I = pop_current(J, dale, rates, src, trg)
            rms[src][trg] = unit_rms(I)
            Itr = reshape_trials(I, n_tr, tlen)
            over_time[src][trg] = Itr.mean(axis=(0, 1)).astype(np.float32)
            per_trial[src][trg] = Itr.mean(axis=0).astype(np.float32)
        Iin = drive[tidx] if tidx.size else np.zeros((1, rates.shape[1]), dtype=np.float32)
        rms['input'][trg] = unit_rms(Iin)
        Itr = reshape_trials(Iin, n_tr, tlen)
        over_time['input'][trg] = Itr.mean(axis=(0, 1)).astype(np.float32)
        per_trial['input'][trg] = Itr.mean(axis=0).astype(np.float32)
    return rms, over_time, per_trial


def pack_condition(condition, rms, over_time, per_trial, meta):
    blob = dict(meta)
    blob.update(
        condition=condition,
        source_order=list(SOURCES),
        target_order=list(POPS),
        influence_rms=rms,
        influence_rms_array=stack_rms(rms),
        currents_over_time=over_time,
        currents_over_time_array=stack_time(over_time, int(meta['trial_length'])),
        currents_per_trial=per_trial,
        note=(
            'Ground-truth CURBD currents for this fitted RNN, not recovered '
            'synapses. This file is {cond} only. influence_rms is aggregated '
            'from these currents (mean over target units of RMS over time of '
            'J[trg,src] @ r_src). currents_over_time is the unit- and '
            'trial-mean on the 20 ms clock. Climbing and pseudo-opto are '
            'never mixed in this file.'
            .format(cond=condition)
        ),
    )
    return blob


def meta_fields(model, model_path, dataset, init_tag, sim, dale, t_ms, n_tr, tlen, shift, dtData, dtFactor):
    g = np.asarray(model['gamma']).reshape(-1)
    n1 = len(model['regions']['region1'])
    return dict(
        session=session_tag_from_dataset(dataset),
        init_tag=init_tag,
        trained_on='climbing_only',
        source_model=os.path.abspath(model_path),
        source_dataset=os.path.abspath(dataset),
        sparse_percent=float(model['params'].get('sparse_percent', np.nan)),
        input_shift_data_bins=int(shift),
        dtData=float(dtData),
        dtFactor=int(dtFactor),
        binsize_ms=float(dtData) * 1000.0,
        n_trials=int(n_tr),
        trial_length=int(tlen),
        t_ms=t_ms,
        optoAmp=float(sim['optoAmp']),
        opto_target_population=str(sim['target_population']),
        opto_n_stim=int(len(sim['target_idx'])),
        opto_stim_onset_s=float(sim['stim_onset_s']),
        opto_dur_s=float(sim['dur']),
        dale_index={k: np.asarray(v, dtype=np.int32) for k, v in dale.items()},
        gamma=g.astype(np.float32),
        counts=dict(
            CFA_E=int(np.sum(g[:n1] > 0)),
            CFA_I=int(np.sum(g[:n1] < 0)),
            RFA_E=int(np.sum(g[n1:] > 0)),
            RFA_I=int(np.sum(g[n1:] < 0)),
        ),
    )


def export_one(model_path, dataset, init_tag, opto_amp, seed):
    model = attach_opto(load_model(model_path), dataset)
    Adata = np.asarray(model['Adata'], dtype=float)
    J = np.asarray(model['J'], dtype=float)
    dale = curbd.dale_populations(model)
    dtFactor = int(model['params']['dtFactor'])
    shift = int(model['params'].get('input_shift_data_bins', 1))
    dtData = float(model.get('dtData', 0.02))
    tlen = int(model['trial_length'])
    n_tr = int(model['n_trials'])
    Ttot = n_tr * tlen
    Adata = Adata[:, :Ttot]
    t_ms = (np.arange(tlen) * dtData * 1000.0).astype(np.float32)
    wn = np.asarray(model['inputWN'], dtype=float)
    climb_drive = sample_drive(wn, None, Ttot, dtFactor, shift)

    sim = curbd.simulate_pseudo_opto_trials(
        model, optoAmp=opto_amp, seed=seed, reuse_wn=True, with_control=False)
    pred = np.asarray(sim['pred_opto'], dtype=float)[:, :Ttot]
    n_rnn = int(sim['RNN_opto'].shape[1])
    np.random.seed(seed + 1)
    extra = curbd._opto_inhib_pulse(
        J.shape[0], n_rnn, sim['target_idx'], sim['stim_mask'], sim['optoAmp'])
    wn_o = wn[:, :n_rnn] if wn.shape[1] >= n_rnn else wn
    opto_drive = sample_drive(wn_o, extra, Ttot, dtFactor, shift)

    meta = meta_fields(
        model, model_path, dataset, init_tag, sim, dale,
        t_ms, n_tr, tlen, shift, dtData, dtFactor)

    climb_rms, climb_t, climb_tr = currents_for_tape(
        J, dale, Adata, climb_drive, n_tr, tlen)
    opto_rms, opto_t, opto_tr = currents_for_tape(
        J, dale, pred, opto_drive, n_tr, tlen)
    return (
        pack_condition('climbing', climb_rms, climb_t, climb_tr, meta),
        pack_condition('pseudo_opto', opto_rms, opto_t, opto_tr, meta),
    )


def stem_for(blob):
    animal_date = animal_date_from_session(blob['session'])
    if blob['condition'] == 'climbing':
        return '{}_climbingOnly_{}_curbdCurrents_daleEI'.format(
            animal_date, blob['init_tag'])
    return '{}_pseudoOptoOnly_{}_amp{}_curbdCurrents_daleEI'.format(
        animal_date, blob['init_tag'], blob['optoAmp'])


def write_export(blob, outdir, copy_dir=None):
    stem = stem_for(blob)
    os.makedirs(outdir, exist_ok=True)
    pkl_path = os.path.join(outdir, stem + '.pickle')
    js_path = os.path.join(outdir, stem + '.json')
    slim = dict(blob)
    slim.pop('currents_per_trial', None)
    with open(pkl_path, 'wb') as f:
        pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(js_path, 'w') as f:
        json.dump(jsonable(slim), f, indent=2)
    print('Wrote', pkl_path)
    print('Wrote', js_path)
    if copy_dir:
        os.makedirs(copy_dir, exist_ok=True)
        for src in (pkl_path, js_path):
            dst = os.path.join(copy_dir, os.path.basename(src))
            with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
                fdst.write(fsrc.read())
            print('Copied', dst)
    return pkl_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=CO9)
    parser.add_argument('--i0sign-model', default=DEFAULT_FITS[0][1])
    parser.add_argument('--half-flip-model', default=DEFAULT_FITS[1][1])
    parser.add_argument('--inits', default='i0init,50_50')
    parser.add_argument('--optoAmp', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    session = session_tag_from_dataset(args.dataset)
    outdir = args.output_dir or os.path.join(
        ROOT, 'outputs', 'dale_label_export', session)
    init_specs = {
        'i0init': args.i0sign_model,
        '50_50': args.half_flip_model,
    }
    wanted = [t.strip() for t in args.inits.split(',') if t.strip()]
    unknown = [t for t in wanted if t not in init_specs]
    if unknown:
        raise ValueError('Unknown init tags {}: choose from {}'.format(
            unknown, list(init_specs)))

    for init_tag in wanted:
        model_path = init_specs[init_tag]
        climb_blob, opto_blob = export_one(
            model_path, args.dataset, init_tag, args.optoAmp, args.seed)
        copy_dir = os.path.dirname(os.path.abspath(model_path))
        write_export(climb_blob, outdir, copy_dir=copy_dir)
        write_export(opto_blob, outdir, copy_dir=copy_dir)
        print('  init={}  Dale {}  sparse_percent={}'.format(
            init_tag, climb_blob['counts'], climb_blob['sparse_percent']))


if __name__ == '__main__':
    main()
