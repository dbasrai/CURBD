#!/usr/bin/env python
"""Export Dale pop-to-pop CURBD currents, sliced to match the activity export.

Fits are climbing-only. Pseudo-opto is a post-hoc amp-5 I-cell pulse on the
fitted J. Climbing and perturb are written as separate files; RMS aggregation
is computed inside each file from that condition's sliced currents only.

Trial clock matches scripts/exportDaleLabeledSpikes.py:

  Climbing: 41-bin bouts cut into 8 full 100 ms TF windows (5 bins x 20 ms);
  leftover last bin dropped. Shape per src→trg: (n_bouts * 8, 5).

  Pseudo-opto: one 200 ms trial per bout (10 bins) from t=0. Default export
  puts the unit input event at bin 1 (20 ms sample). --input-event-bin 0
  instead labels the 0-20 ms injection interval. Shape per src→trg: (n_bouts, 10).

  {animal}_{date}_climbingOnly_{init}_curbdCurrents_daleEI.pickle
  {animal}_{date}_pseudoOptoOnly_{init}_amp{amp}_curbdCurrents_daleEI.pickle

Each pickle
-----------
condition : 'climbing' or 'pseudo_opto'

influence_rms[source][target] : float
    Mean over target units of the sliced-term RMS. Cross-population terms are
    J[target, source] @ r_source. A population's self term is
    J[target, target] @ r_target - x_target, where r = tanh(x).
    The input source is the designed pseudo-opto event only: zero for climbing
    and one unit-valued binary pulse for pseudo-opto (default data bin 1;
    bin 0 if --input-event-bin 0).
    Background WN is not part of the exported ground-truth intervention.

currents_over_time[source][target] : (T,) float32
    Unit- and trial-mean dynamical term versus time on the sliced clock.

currents_per_trial[source][target] : (n_trials, T) float32
    Same current, mean over target units, trials kept.

Sources: CFA_E, CFA_I, RFA_E, RFA_I, input.
Targets: CFA_E, CFA_I, RFA_E, RFA_I.
"""
from __future__ import print_function

import argparse
import json
import math
import os
import pickle
import sys

import numpy as np
import numpy.random as npr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import curbd
import exportDaleLabeledSpikes as spike_export

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


def make_input_wn(model, n_rnn, seed):
    """Same filtered WN as simulate_pseudo_opto_trials when reuse_wn is False."""
    params = model['params']
    dtData = float(model.get('dtData', params.get('dtData', 0.02)))
    dtFactor = int(params['dtFactor'])
    dtRNN = float(model.get('dtRNN', dtData / float(dtFactor)))
    npr.seed(seed)
    tauWN = float(params.get('tauWN', 0.1))
    ampInWN = float(params.get('ampInWN', 0.001))
    ampWN = math.sqrt(tauWN / dtRNN)
    n_units = int(params['number_units'])
    iWN = ampWN * npr.randn(n_units, n_rnn)
    wn = np.ones((n_units, n_rnn))
    for tt in range(1, n_rnn):
        wn[:, tt] = iWN[:, tt] + (wn[:, tt - 1] - iWN[:, tt]) * np.exp(
            -(dtRNN / tauWN))
    return ampInWN * wn


def move_last_axis_event(arr, src_bin, dst_bin):
    """Move a one-hot along the last axis. Other axes (units, trials) unchanged."""
    out = np.array(arr, dtype=np.float32, copy=True)
    src_bin = int(src_bin)
    dst_bin = int(dst_bin)
    if src_bin == dst_bin:
        return out
    if src_bin < 0 or dst_bin < 0 or src_bin >= out.shape[-1] or dst_bin >= out.shape[-1]:
        raise ValueError('event bins {} -> {} out of range for T={}'.format(
            src_bin, dst_bin, out.shape[-1]))
    pulse = np.array(out[..., src_bin], copy=True)
    out[..., src_bin] = 0
    out[..., dst_bin] = pulse
    return out


def retarget_bout_events(drive, n_tr, tlen, target_idx, src_bin, dst_bin):
    """Move per-bout 1-hots on the unsliced (N, n_trials * T) drive."""
    out = np.array(drive, dtype=np.float32, copy=True)
    target_idx = np.asarray(target_idx, dtype=int)
    src_bin = int(src_bin)
    dst_bin = int(dst_bin)
    if src_bin == dst_bin or target_idx.size == 0:
        return out
    for tr in range(int(n_tr)):
        src = tr * int(tlen) + src_bin
        dst = tr * int(tlen) + dst_bin
        if src >= out.shape[1] or dst >= out.shape[1]:
            continue
        pulse = np.array(out[target_idx, src], copy=True)
        out[target_idx, src] = 0
        out[target_idx, dst] = pulse
    return out


def relabel_current_input_event(blob, src_bin, dst_bin):
    """Rewrite only the exported input 1-hot. J r terms are left untouched."""
    src_bin = int(src_bin)
    dst_bin = int(dst_bin)
    out = dict(blob)
    if src_bin == dst_bin:
        return out
    pops = list(out.get('target_order', POPS))
    sources = list(out.get('source_order', SOURCES))
    over = out.get('currents_over_time')
    if over is not None and 'input' in over:
        over_in = dict(over)
        over_in['input'] = dict(over['input'])
        for trg in pops:
            over_in['input'][trg] = move_last_axis_event(
                over['input'][trg], src_bin, dst_bin)
        out['currents_over_time'] = over_in
    per = out.get('currents_per_trial')
    if per is not None and 'input' in per:
        per_in = dict(per)
        per_in['input'] = dict(per['input'])
        for trg in pops:
            per_in['input'][trg] = move_last_axis_event(
                per['input'][trg], src_bin, dst_bin)
        out['currents_per_trial'] = per_in
    arr = out.get('currents_over_time_array')
    if arr is not None and 'input' in sources:
        arr = np.array(arr, dtype=np.float32, copy=True)
        i_src = sources.index('input')
        arr[i_src] = move_last_axis_event(arr[i_src], src_bin, dst_bin)
        out['currents_over_time_array'] = arr
    out['stim_bin'] = dst_bin
    out['input_event_bin'] = dst_bin
    out['sampled_response_bin'] = src_bin
    note = str(out.get('note', ''))
    note = note.replace(
        'unit-valued binary pulse at bin 1',
        'unit-valued binary pulse at bin {}'.format(dst_bin))
    out['note'] = note
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


def unit_currents_bouts(J, dale, rates, hidden, drive, n_tr, tlen):
    """Population terms on the native bout clock.

    For source != target, the term is J[target, source] @ r_source.
    For source == target, -x_target is merged into the recurrent same-pop
    term so all exported terms sum to the RNN equation's right-hand side.
    """
    Ttot = n_tr * tlen
    rates = np.asarray(rates, dtype=float)[:, :Ttot]
    hidden = np.asarray(hidden, dtype=float)[:, :Ttot]
    drive = np.asarray(drive, dtype=np.float32)[:, :Ttot]
    out = nest_sources_targets(lambda: None)
    for trg in POPS:
        tidx = np.asarray(dale[trg], dtype=int)
        for src in POPS:
            I = pop_current(J, dale, rates, src, trg)
            if src == trg:
                I = I - hidden[tidx]
            out[src][trg] = reshape_trials(I, n_tr, tlen)
        Iin = drive[tidx] if tidx.size else np.zeros((1, Ttot), dtype=np.float32)
        out['input'][trg] = reshape_trials(Iin, n_tr, tlen)
    return out


def map_nested_bouts(nested, fn):
    """Apply fn(I_bouts) -> (I_sliced, behav, start) to each src→trg block."""
    sliced = nest_sources_targets(lambda: None)
    behav = start_bin = None
    for src in SOURCES:
        for trg in POPS:
            cut, b, s = fn(nested[src][trg])
            sliced[src][trg] = cut
            behav, start_bin = b, s
    return sliced, behav, start_bin


def aggregate_sliced(nested):
    """RMS / PSTH / per-trial from sliced (n_units, n_trials, T) blocks."""
    rms = nest_sources_targets(lambda: 0.0)
    over_time = nest_sources_targets(lambda: None)
    per_trial = nest_sources_targets(lambda: None)
    n_tr = T = None
    for src in SOURCES:
        for trg in POPS:
            I = nested[src][trg]
            n_u, n_tr, T = I.shape
            rms[src][trg] = unit_rms(I.reshape(n_u, n_tr * T))
            over_time[src][trg] = I.mean(axis=(0, 1)).astype(np.float32)
            per_trial[src][trg] = I.mean(axis=0).astype(np.float32)
    return rms, over_time, per_trial, int(n_tr), int(T)


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
            'synapses. This file is {cond} only, sliced to the same trial clock '
            'as the Dale-labeled activity export. influence_rms is aggregated '
            'from these sliced currents (mean over target units of RMS over '
            'time). Cross-population terms are J[trg,src] @ r_src. Each '
            'same-population self term merges intrinsic decay as '
            'J[trg,trg] @ r_trg - x_trg, with r=tanh(x). Therefore the '
            'population terms are on the unscaled right-hand-side current '
            'scale of tau*dx/dt. The exported input is the deterministic '
            'experimental event indicator only: zero for climbing and a '
            'unit-valued binary pulse at bin {bin} for pseudo-opto. Its amplitude '
            'is independent of the physical optoAmp used to generate activity. '
            'Background filtered WN used '
            'during simulation is intentionally excluded from this GT input. '
            'currents_over_time is the unit- and trial-mean on the 20 ms '
            'sliced clock. Climbing and pseudo-opto are never mixed.'
            .format(cond=condition, bin=int(meta.get('stim_bin', 1)))
        ),
    )
    return blob


def shared_meta(model, model_path, dataset, init_tag, dale, shift, dtData,
                dtFactor, reset_every, n_behav, tlen_bout, n_fit_behav):
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
        tauRNN=float(model['params']['tauRNN']),
        dynamics_scale='unscaled_rhs_of_tau_dxdt',
        self_term_includes_decay=True,
        self_term_definition='J[target,target] @ r_target - x_target',
        hidden_from_rate='arctanh(clip(r, -0.999, 0.999))',
        input_definition='unit-valued binary intervention event',
        input_units='binary_indicator',
        background_wn_exported_as_input=False,
        reset_every_rnn=int(reset_every),
        n_behavioral_trials=int(n_behav),
        n_fit_behavioral_trials=int(n_fit_behav),
        behavioral_trial_length=int(tlen_bout),
        dale_index={k: np.asarray(v, dtype=np.int32) for k, v in dale.items()},
        gamma=g.astype(np.float32),
        counts=dict(
            CFA_E=int(np.sum(g[:n1] > 0)),
            CFA_I=int(np.sum(g[:n1] < 0)),
            RFA_E=int(np.sum(g[n1:] > 0)),
            RFA_I=int(np.sum(g[n1:] < 0)),
        ),
    )


def condition_meta(base, sim, t_ms, n_tr, tlen, behav, start_bin, win_starts,
                   n_fit_behav, extra=None):
    fit_seg = np.asarray(behav) < int(n_fit_behav)
    out = dict(base)
    out.update(
        n_trials=int(n_tr),
        trial_length=int(tlen),
        t_ms=np.asarray(t_ms, dtype=np.float32),
        tf_bins=int(tlen),
        tf_ms=int(round(tlen * float(base['binsize_ms']))),
        n_fit_trials=int(np.sum(fit_seg)),
        behavioral_trial=np.asarray(behav, dtype=np.int32),
        tf_start_bin=np.asarray(start_bin, dtype=np.int32),
        tf_window_starts_in_trial=np.asarray(win_starts, dtype=np.int32),
        optoAmp=float(sim['optoAmp']),
        opto_target_population=str(sim['target_population']),
        opto_n_stim=int(len(sim['target_idx'])),
        opto_stim_onset_s=float(sim['stim_onset_s']),
        opto_dur_s=float(sim['dur']),
    )
    if extra:
        out.update(extra)
    return out


def export_one(model_path, dataset, init_tag, opto_amp, seed, fit_trials=40,
               opto_ms=200, reset_every=25, input_event_bin=None):
    model = load_model(model_path)
    dtFactor = int(model['params']['dtFactor'])
    shift = int(model['params'].get('input_shift_data_bins', 1))
    dtData = float(model.get('dtData', 0.02))
    data = curbd.load_ei_dataset(
        dataset, dtFactor=dtFactor, smooth_sigma=1.5, zscore=False)
    model = spike_export.prepare_opto_sim_model(model, data)
    dale = curbd.dale_populations(model)
    J = np.asarray(model['J'], dtype=float)
    n_bouts = int(data['n_trials'])
    tlen_bout = int(data['trial_length'])
    n_fit_behav = min(int(fit_trials), n_bouts)
    Ttot = n_bouts * tlen_bout
    Adata = np.asarray(model['Adata'], dtype=float)[:, :Ttot]
    nonLinearity_inv = model['params'].get('nonLinearity_inv', np.arctanh)
    climb_hidden = nonLinearity_inv(np.clip(Adata, -0.999, 0.999))
    n_rnn = Ttot * dtFactor
    wn = make_input_wn(model, n_rnn, seed)
    model['inputWN'] = wn
    climb_drive = np.zeros((J.shape[0], Ttot), dtype=np.float32)

    print('Simulating pseudo-opto init={} amp={}  {} bouts'.format(
        init_tag, opto_amp, n_bouts))
    sim = curbd.simulate_pseudo_opto_trials(
        model, n_trials=n_bouts, trial_length=tlen_bout,
        optoAmp=opto_amp, seed=seed, reuse_wn=True, with_control=False)
    pred = np.asarray(sim['pred_opto'], dtype=float)[:, :Ttot]
    opto_hidden = nonLinearity_inv(np.clip(pred, -0.999, 0.999))
    opto_drive = np.asarray(sim['opto_input_data'], dtype=np.float32)[:, :Ttot]
    event_bin = int(sim['stim_bin']) if input_event_bin is None else int(input_event_bin)
    if event_bin != int(sim['stim_bin']):
        opto_drive = retarget_bout_events(
            opto_drive, n_bouts, tlen_bout, sim['target_idx'],
            int(sim['stim_bin']), event_bin)

    climb_bouts = unit_currents_bouts(
        J, dale, Adata, climb_hidden, climb_drive, n_bouts, tlen_bout)
    opto_bouts = unit_currents_bouts(
        J, dale, pred, opto_hidden, opto_drive, n_bouts, tlen_bout)

    starts, tf_bins = spike_export.tf_window_bins(tlen_bout, dtFactor, reset_every)
    climb_sliced, climb_behav, climb_start = map_nested_bouts(
        climb_bouts, lambda I: spike_export.cut_windows(I, starts, tf_bins))
    opto_bins, expected_stim_bin = spike_export.opto_trial_bins(
        opto_ms=opto_ms, binsize_ms=float(dtData) * 1000.0,
        stim_onset_ms=float(sim['stim_onset_s']) * 1000.0)
    if int(sim['stim_bin']) != int(expected_stim_bin):
        raise ValueError('Simulator stim bin {} != export stim bin {}'.format(
            sim['stim_bin'], expected_stim_bin))
    opto_sliced, opto_behav, opto_start = map_nested_bouts(
        opto_bouts, lambda I: spike_export.slice_opto_trials(I, opto_bins))

    climb_rms, climb_t, climb_tr, n_climb, T_climb = aggregate_sliced(climb_sliced)
    opto_rms, opto_t, opto_tr, n_opto, T_opto = aggregate_sliced(opto_sliced)

    base = shared_meta(
        model, model_path, dataset, init_tag, dale, shift, dtData,
        dtFactor, reset_every, n_bouts, tlen_bout, n_fit_behav)
    climb_t_ms = (np.arange(T_climb) * dtData * 1000.0).astype(np.float32)
    opto_t_ms = (np.arange(T_opto) * dtData * 1000.0).astype(np.float32)
    climb_meta = condition_meta(
        base, sim, climb_t_ms, n_climb, T_climb, climb_behav, climb_start,
        starts, n_fit_behav,
        extra=dict(
            source='climbing_recorded',
            leftover_bin_dropped=True,
        ))
    opto_meta = condition_meta(
        base, sim, opto_t_ms, n_opto, T_opto, opto_behav, opto_start,
        np.array([0], dtype=np.int32), n_fit_behav,
        extra=dict(
            source='pseudo_opto',
            stim_bin=int(event_bin),
            input_event_bin=int(event_bin),
            sampled_response_bin=int(sim['stim_bin']),
            opto_window_ms=[0, int(opto_ms)],
            input_is_binary_pulse=True,
            input_pulse_amplitude=1.0,
            physical_opto_amplitude=float(opto_amp),
        ))
    return (
        pack_condition('climbing', climb_rms, climb_t, climb_tr, climb_meta),
        pack_condition('pseudo_opto', opto_rms, opto_t, opto_tr, opto_meta),
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
    parser.add_argument('--fit-trials', type=int, default=40,
                        help='First N behavioral trials used to fit the Dale models')
    parser.add_argument('--opto-ms', type=float, default=200,
                        help='Pseudo-opto trial length from t=0 (stim at 20 ms = bin 1)')
    parser.add_argument('--input-event-bin', type=int, default=None,
                        help='Data-bin of the exported unit input 1-hot. '
                             'Default is the stim-onset bin (1). 0 labels the '
                             '0-20 ms injection interval. Does not change J or rates.')
    parser.add_argument('--reset-every', type=int, default=25,
                        help='Teacher-force period in RNN steps (25 = 100 ms at dtFactor=5)')
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
            model_path, args.dataset, init_tag, args.optoAmp, args.seed,
            fit_trials=args.fit_trials, opto_ms=args.opto_ms,
            reset_every=args.reset_every, input_event_bin=args.input_event_bin)
        copy_dir = None if args.input_event_bin is not None else os.path.dirname(
            os.path.abspath(model_path))
        write_export(climb_blob, outdir, copy_dir=copy_dir)
        write_export(opto_blob, outdir, copy_dir=copy_dir)
        print('  init={}  Dale {}  sparse_percent={}  climbing {} x {}  opto {} x {}'.format(
            init_tag, climb_blob['counts'], climb_blob['sparse_percent'],
            climb_blob['n_trials'], climb_blob['trial_length'],
            opto_blob['n_trials'], opto_blob['trial_length']))


if __name__ == '__main__':
    main()
