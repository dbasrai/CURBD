#!/usr/bin/env python
"""Export recorded spike counts plus final Dale E/I labels, and optional
pseudo-opto trials merged with climbing.

Climbing: 41-bin bouts cut into 8 full 100 ms TF windows (5 bins x 20 ms);
the leftover last bin is dropped.

Pseudo-opto: deterministic I-cell pulse (onset at 20 ms). One 200 ms trial
per bout (10 bins) from t=0. Default metadata marks stim at bin 1; use
--input-event-bin 0 to label the 0-20 ms injection interval instead.
Spikes/rates are unchanged.

Merged pickles right-pad climbing to 10 bins with NaN and a valid_mask.
Do not treat padded bins as real zeros in a loss.

Filenames include init (50_50 or i0init). Opto/merged also include amp:
  {animal}_{date}_climbingOnly_{init}_spikeCounts_daleEI.pickle
  {animal}_{date}_pseudoOptoOnly_{init}_amp{amp}_spikeCounts_daleEI.pickle
  {animal}_{date}_climbingPlusPseudoOpto_{init}_amp{amp}_spikeCounts_daleEI.pickle
"""
from __future__ import print_function

import argparse
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
HALF_FLIP = os.path.join(
    ROOT, 'outputs', 'dale_horizon_balance',
    'co9_DALE_i0rank_half_flip_reset25', 'model.pickle')
I0_SIGN = os.path.join(
    ROOT, 'outputs', 'dale_scalar_co9',
    'DALE_i0sign_reset25', 'model.pickle')


def session_tag_from_dataset(pkl_path):
    """e.g. co9_12122023_climbing from the aligned pickle stem."""
    stem = os.path.splitext(os.path.basename(pkl_path))[0]
    parts = stem.split('_')
    if len(parts) >= 3:
        return '_'.join(parts[:3])
    return stem


def animal_date_from_session(session):
    """co9_12122023_climbing -> co9_12122023."""
    parts = str(session).split('_')
    if len(parts) >= 2:
        return '_'.join(parts[:2])
    return str(session)


def amp_tag(amp):
    return 'amp{}'.format(float(amp))


def export_name_stems(session, init_tag, opto_amp=None):
    """Filename stems: init tag always; amp when opto was used."""
    animal_date = animal_date_from_session(session)
    stems = {
        'climbing': '{}_climbingOnly_{}'.format(animal_date, init_tag),
    }
    if opto_amp is not None:
        a = amp_tag(opto_amp)
        stems['opto'] = '{}_pseudoOptoOnly_{}_{}'.format(
            animal_date, init_tag, a)
        stems['merged'] = '{}_climbingPlusPseudoOpto_{}_{}'.format(
            animal_date, init_tag, a)
    return stems


def spike_pickle_path(outdir, stem):
    return os.path.join(outdir, '{}_spikeCounts_daleEI.pickle'.format(stem))


def load_model(path):
    with open(path, 'rb') as f:
        blob = pickle.load(f)
    if isinstance(blob, dict) and 'model' in blob:
        return blob['model']
    if isinstance(blob, dict) and 'J' in blob:
        return blob
    raise ValueError('Unrecognized model pickle: {}'.format(path))


def load_raw_spikes(pkl_path):
    """Integer counts, concat order CFA_E, CFA_I, RFA_E, RFA_I.

    Returns spikes (N, n_trials, T), pickle_pop names (N,), pickle_label (N,).
    """
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)
    blocks = []
    pop_names = []
    labels = []
    n_trials = None
    trial_len = None
    for name in curbd.EI_POP_ORDER:
        trials = [np.asarray(t) for t in raw[name]]
        arr = np.stack(trials, axis=0)
        if n_trials is None:
            n_trials, trial_len, _ = arr.shape
        elif arr.shape[0] != n_trials or arr.shape[1] != trial_len:
            raise ValueError('Trial shape mismatch for {}'.format(name))
        n_u = arr.shape[2]
        blocks.append(arr)
        pop_names.extend([name] * n_u)
        labels.extend(['E' if name.endswith('_E') else 'I'] * n_u)
    spikes = np.concatenate(blocks, axis=2).transpose(2, 0, 1)
    if not np.allclose(spikes, np.round(spikes)):
        raise ValueError('Expected integer spike counts in {}'.format(pkl_path))
    return spikes.astype(np.int16), np.asarray(pop_names), np.asarray(labels)


def dale_fields(model):
    g = np.asarray(model['gamma']).reshape(-1)
    g0 = np.asarray(model.get('gamma0', g)).reshape(-1)
    n1 = len(model['regions']['region1'])
    lab = np.where(g > 0, 'E', 'I')
    lab0 = np.where(g0 > 0, 'E', 'I')
    region = np.array(['CFA'] * n1 + ['RFA'] * (g.size - n1))
    return dict(gamma=g, gamma0=g0, label=lab, label0=lab0, region=region, n1=n1)


def tf_window_bins(trial_length, dtFactor, reset_every):
    """Data-bin starts of complete teacher-force windows.

    reset_every is in RNN steps. One data bin is dtFactor RNN steps, so a
    100 ms TF with 20 ms bins is reset_every=25, dtFactor=5, tf_bins=5.
    A 41-bin climbing trial yields windows at 0,5,...,35 (8 windows);
    the leftover last bin is dropped because it is not a full TF interval.
    """
    tf_bins = int(reset_every) // int(dtFactor)
    if tf_bins < 1 or int(reset_every) % int(dtFactor) != 0:
        raise ValueError(
            'reset_every={} is not a whole number of data bins (dtFactor={})'
            .format(reset_every, dtFactor))
    starts = np.arange(0, int(trial_length) - tf_bins + 1, tf_bins, dtype=np.int32)
    return starts, tf_bins


def cut_windows(spikes, starts, tf_bins):
    """Cut (N, n_behav, T) into (N, n_behav * n_win, tf_bins)."""
    n, n_behav, T = spikes.shape
    starts = np.asarray(starts, dtype=np.int32)
    n_win = len(starts)
    if n_win == 0:
        raise ValueError('No windows to cut')
    if int(starts[-1]) + int(tf_bins) > T:
        raise ValueError('Window {}+{} exceeds trial length {}'.format(
            starts[-1], tf_bins, T))
    out = np.empty((n, n_behav * n_win, int(tf_bins)), dtype=spikes.dtype)
    behav = np.empty(n_behav * n_win, dtype=np.int32)
    start_bin = np.empty(n_behav * n_win, dtype=np.int32)
    k = 0
    for tr in range(n_behav):
        for s in starts:
            out[:, k, :] = spikes[:, tr, int(s):int(s) + int(tf_bins)]
            behav[k] = tr
            start_bin[k] = s
            k += 1
    return out, behav, start_bin


def cut_tf_trials(spikes, dtFactor=5, reset_every=25):
    """Reshape (N, n_behav_trials, T) into (N, n_tf_trials, tf_bins)."""
    T = spikes.shape[2]
    starts, tf_bins = tf_window_bins(T, dtFactor, reset_every)
    out, behav, start_bin = cut_windows(spikes, starts, tf_bins)
    return out, behav, start_bin, tf_bins, starts


def opto_trial_bins(opto_ms=200, binsize_ms=20, stim_onset_ms=20):
    """One trial from t=0, long enough to include stim at timepoint 1.

    20 ms bins: bin 0 is 0–20 ms, stim at 20 ms is bin 1, 200 ms is 10 bins.
    """
    n_bins = int(round(float(opto_ms) / float(binsize_ms)))
    stim_bin = int(round(float(stim_onset_ms) / float(binsize_ms)))
    if n_bins < stim_bin + 1:
        raise ValueError('opto_ms={} is shorter than stim at {} ms'.format(
            opto_ms, stim_onset_ms))
    return n_bins, stim_bin


def slice_opto_trials(spikes_bouts, n_bins):
    """Keep bins [0, n_bins) as one trial per bout."""
    n, n_bouts, T = spikes_bouts.shape
    if T < int(n_bins):
        raise ValueError('Bout length {} < opto bins {}'.format(T, n_bins))
    out = spikes_bouts[:, :, :int(n_bins)]
    behav = np.arange(n_bouts, dtype=np.int32)
    start_bin = np.zeros(n_bouts, dtype=np.int32)
    return out, behav, start_bin


def pad_trials_right(spikes, target_bins):
    """Right-pad (N, n_trials, T) with NaN. Mask is (n_trials, target_bins)."""
    n, n_tr, T = spikes.shape
    target_bins = int(target_bins)
    if T > target_bins:
        raise ValueError('Trial length {} exceeds pad width {}'.format(
            T, target_bins))
    padded = np.full((n, n_tr, target_bins), np.nan, dtype=np.float32)
    padded[:, :, :T] = spikes
    mask = np.zeros((n_tr, target_bins), dtype=bool)
    mask[:, :T] = True
    lengths = np.full(n_tr, T, dtype=np.int32)
    return padded, mask, lengths


def prepare_opto_sim_model(model, data):
    """Full-session Adata in the fit's tanh scale; new WN for the longer tape.

    `data['z_activity']` should be smoothed but not z-scored so we can apply
    the model's training scaler.
    """
    model = dict(model)
    activity = np.asarray(data['z_activity'], dtype=float)
    scaler = model.get('scaler') or data.get('scaler')
    train_max = float(model.get('train_max') or np.max(np.abs(model['Adata'])) or 1.0)
    if scaler is not None and hasattr(scaler, 'transform'):
        z = scaler.transform(activity.T).T
    elif scaler is not None and hasattr(scaler, 'mean_'):
        z = ((activity.T - scaler.mean_) / np.where(scaler.scale_ == 0, 1.0, scaler.scale_)).T
    else:
        z = activity
    model['Adata'] = np.clip(z / train_max, -0.999, 0.999)
    model['train_max'] = train_max
    model['scaler'] = scaler
    model['populations'] = data['populations']
    model['pkl_path'] = data['pkl_path']
    model['opto_target_population'] = data.get('opto_target_population')
    model['opto_corresponding_e_population'] = data.get(
        'opto_corresponding_e_population')
    model['stimulated_region'] = data.get('stimulated_region')
    model['stim_onset_s'] = data.get('stim_onset_s', 0.02)
    model['n_trials'] = data['n_trials']
    model['trial_length'] = data['trial_length']
    model['inputWN'] = None
    return model


def tanh_rates_to_counts(rates, scaler, train_max, poisson_mult=1.0, seed=0):
    """Map tanh rates back toward binned counts (same scale as recorded spikes)."""
    rng = np.random.RandomState(seed)
    rates_tn = np.asarray(rates, dtype=float).T
    lam = scaler.inverse_transform(rates_tn * float(train_max)) * float(poisson_mult)
    lam = np.maximum(lam, 0.0)
    spikes_tn = rng.poisson(lam).astype(np.int16)
    return spikes_tn.T


def regroup_trial_lists(spikes, region, label, trial_lengths=None):
    """CURBD loader layout: dict of pop -> list of (T, n_pop) trials.

    If trial_lengths is set, each trial is cropped to its native T so padded
    NaNs are not stored as data.
    """
    out = {}
    n_trials = spikes.shape[1]
    if trial_lengths is None:
        trial_lengths = np.full(n_trials, spikes.shape[2], dtype=np.int32)
    for pop in curbd.EI_POP_ORDER:
        reg, lab = pop.split('_')
        idx = np.where((region == reg) & (label == lab))[0]
        out[pop] = [
            np.asarray(spikes[idx, tr, :int(trial_lengths[tr])].T)
            for tr in range(n_trials)
        ]
    return out


def counts_for(model):
    g = np.asarray(model['gamma']).reshape(-1)
    n1 = len(model['regions']['region1'])
    return dict(
        n=int(g.size),
        n_E=int(np.sum(g > 0)),
        n_I=int(np.sum(g < 0)),
        CFA_E=int(np.sum(g[:n1] > 0)),
        CFA_I=int(np.sum(g[:n1] < 0)),
        RFA_E=int(np.sum(g[n1:] > 0)),
        RFA_I=int(np.sum(g[n1:] < 0)),
        n_flip=int(np.sum(np.sign(g) != np.sign(
            np.asarray(model['gamma0']).reshape(-1)))),
    )


def pack_spike_export(
        session, spikes_tf, behav_trial, tf_start_bin, win_starts, tf_bins,
        pickle_pop, pickle_lab, dale, dale_model, dale_path, init_tag, dataset,
        n_behav, trial_len, n_fit_behav, dtFactor, reset_every, condition,
        extra=None, note='', trial_lengths=None, valid_mask=None):
    n, n_tf, T = spikes_tf.shape
    if trial_lengths is None:
        trial_lengths = np.full(n_tf, int(tf_bins), dtype=np.int32)
    else:
        trial_lengths = np.asarray(trial_lengths, dtype=np.int32)
    if valid_mask is None:
        valid_mask = np.zeros((n_tf, T), dtype=bool)
        for i, L in enumerate(trial_lengths):
            valid_mask[i, :int(L)] = True
    fit_seg = behav_trial < n_fit_behav
    n_spikes = np.nansum(spikes_tf, axis=(1, 2)).astype(np.int64)
    n_spikes_fit = np.nansum(spikes_tf[:, fit_seg, :], axis=(1, 2)).astype(np.int64)
    unique_L = np.unique(trial_lengths)
    out = dict(
        session=session,
        init_tag=init_tag,
        spikes=spikes_tf,
        valid_mask=valid_mask,
        trial_length=int(unique_L[0]) if unique_L.size == 1 else unique_L,
        trial_lengths=trial_lengths,
        n_spikes=n_spikes,
        n_spikes_fit_trials=n_spikes_fit,
        binsize_ms=20,
        dtData=0.02,
        dtFactor=int(dtFactor),
        reset_every_rnn=int(reset_every),
        tf_bins=int(tf_bins),
        tf_ms=int(tf_bins) * 20,
        n_units=int(n),
        n_trials=int(n_tf),
        n_fit_trials=int(np.sum(fit_seg)),
        n_behavioral_trials=int(n_behav),
        n_fit_behavioral_trials=int(n_fit_behav),
        behavioral_trial_length=int(trial_len),
        behavioral_trial=behav_trial,
        tf_start_bin=tf_start_bin,
        tf_window_starts_in_trial=np.asarray(win_starts, dtype=np.int32),
        condition=np.asarray(condition),
        source_dataset=os.path.abspath(dataset),
        pickle_population=pickle_pop,
        pickle_label=pickle_lab,
        region=dale['region'],
        labels=dict(
            init=dale['label0'],
            final=dale['label'],
            gamma=dale['gamma'],
        ),
        trial_lists=regroup_trial_lists(
            spikes_tf, dale['region'], dale['label'], trial_lengths),
        source_model=os.path.abspath(dale_path),
        counts=counts_for(dale_model),
        note=note,
    )
    if extra:
        out.update(extra)
    return out


def relabel_spike_input_event(blob, src_bin, dst_bin):
    """Rewrite stim_bin metadata only. Spike counts stay put."""
    src_bin = int(src_bin)
    dst_bin = int(dst_bin)
    out = dict(blob)
    if src_bin == dst_bin:
        return out
    if out.get('stim_bin') is None:
        return out
    out['stim_bin'] = dst_bin
    out['input_event_bin'] = dst_bin
    out['sampled_response_bin'] = src_bin
    note = str(out.get('note', ''))
    note = note.replace(
        'stim at timepoint {}'.format(src_bin),
        'input event at timepoint {}'.format(dst_bin))
    note = note.replace(
        'stim at bin {}'.format(src_bin),
        'input event at bin {}'.format(dst_bin))
    note = note.replace(' (20 ms)', ' (0-20 ms injection interval)')
    out['note'] = note
    return out


def write_pickle(path, blob):
    with open(path, 'wb') as f:
        pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Wrote', path)
    sp = blob['spikes']
    lengths = blob.get('trial_lengths')
    length_note = ''
    if lengths is not None:
        uniq, cnt = np.unique(lengths, return_counts=True)
        length_note = '  T={}'.format(
            dict(zip([int(x) for x in uniq], [int(c) for c in cnt])))
    print('  spikes {} x {} x {}{}  conditions: {}'.format(
        sp.shape[0], sp.shape[1], sp.shape[2], length_note,
        {k: int(np.sum(blob['condition'] == k))
         for k in np.unique(blob['condition'])}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=CO9)
    parser.add_argument('--half-flip-model', default=HALF_FLIP)
    parser.add_argument('--i0sign-model', default=I0_SIGN)
    parser.add_argument('--inits', default='50_50,i0init',
                        help='Comma-separated init tags to export (50_50, i0init)')
    parser.add_argument('--optoAmp', type=float, default=5.0)
    parser.add_argument('--opto-ms', type=float, default=200,
                        help='Pseudo-opto trial length from t=0 (stim at 20 ms = bin 1)')
    parser.add_argument('--input-event-bin', type=int, default=None,
                        help='Metadata bin of the exported input event. '
                             'Default is stim-onset bin 1. 0 labels the '
                             '0-20 ms injection interval. Spike counts unchanged.')
    parser.add_argument('--poisson-mult', type=float, default=1.0,
                        help='Scale reconstructed rates before Poisson (1 = count scale)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-opto', action='store_true',
                        help='Only write the climbing TF-window pickle')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--fit-trials', type=int, default=40,
                        help='First N behavioral trials used to fit the Dale models')
    parser.add_argument('--dtFactor', type=int, default=5)
    parser.add_argument('--reset-every', type=int, default=25,
                        help='Teacher-force period in RNN steps (25 = 100 ms at dtFactor=5)')
    args = parser.parse_args()

    session = session_tag_from_dataset(args.dataset)
    if args.output_dir is None:
        args.output_dir = os.path.join(
            ROOT, 'outputs', 'dale_label_export', session)

    spikes, pickle_pop, pickle_lab = load_raw_spikes(args.dataset)
    n, n_trials, trial_len = spikes.shape
    half = load_model(args.half_flip_model)
    i0s = load_model(args.i0sign_model)
    h = dale_fields(half)
    s = dale_fields(i0s)
    if h['gamma'].size != n or s['gamma'].size != n:
        raise ValueError('Unit count mismatch: spikes {} half {} i0sign {}'.format(
            n, h['gamma'].size, s['gamma'].size))
    if not np.all(h['region'] == s['region']):
        raise ValueError('Region split differs between models')

    n_fit_behav = min(int(args.fit_trials), n_trials)
    spikes_tf, behav_trial, tf_start_bin, tf_bins, win_starts = cut_tf_trials(
        spikes, dtFactor=args.dtFactor, reset_every=args.reset_every)
    tf_ms = int(tf_bins) * 20
    cond_climb = np.array(['climbing'] * spikes_tf.shape[1])
    init_specs = {
        '50_50': dict(model=half, fields=h, path=args.half_flip_model),
        'i0init': dict(model=i0s, fields=s, path=args.i0sign_model),
    }
    wanted = [t.strip() for t in args.inits.split(',') if t.strip()]
    unknown = [t for t in wanted if t not in init_specs]
    if unknown:
        raise ValueError('Unknown init tags {}: choose from {}'.format(
            unknown, list(init_specs)))

    os.makedirs(args.output_dir, exist_ok=True)
    data = None if args.no_opto else curbd.load_ei_dataset(
        args.dataset, dtFactor=args.dtFactor, smooth_sigma=1.5, zscore=False)

    for init_tag in wanted:
        spec = init_specs[init_tag]
        pack_kw = dict(
            pickle_pop=pickle_pop, pickle_lab=pickle_lab,
            dale=spec['fields'], dale_model=spec['model'],
            dale_path=spec['path'], init_tag=init_tag, dataset=args.dataset,
            n_behav=n_trials, trial_len=trial_len, n_fit_behav=n_fit_behav,
            dtFactor=args.dtFactor, reset_every=args.reset_every,
        )
        stems = export_name_stems(
            session, init_tag, None if args.no_opto else args.optoAmp)
        cts = counts_for(spec['model'])
        climb = pack_spike_export(
            stems['climbing'], spikes_tf, behav_trial, tf_start_bin,
            win_starts, tf_bins, condition=cond_climb,
            extra=dict(source='climbing_recorded', init_tag=init_tag),
            note=(
                'Climbing-only ({init}): each spikes[:, trial, :] is one '
                'teacher-force window: {bins} bins x 20 ms = {ms} ms. '
                'Windows start at {starts}; leftover last bin dropped.'
                .format(init=init_tag, bins=tf_bins, ms=tf_ms,
                        starts=list(map(int, win_starts)))
            ),
            **pack_kw)
        write_pickle(spike_pickle_path(args.output_dir, stems['climbing']), climb)
        print('  init={}  E/I {}/{}  {} windows/climbing-trial'.format(
            init_tag, cts['n_E'], cts['n_I'], len(win_starts)))
        if args.no_opto:
            continue

        opto_model = prepare_opto_sim_model(spec['model'], data)
        print('Simulating pseudo-opto init={} target={} amp={}  {} bouts'.format(
            init_tag, opto_model.get('opto_target_population'), args.optoAmp,
            data['n_trials']))
        sim = curbd.simulate_pseudo_opto_trials(
            opto_model, n_trials=data['n_trials'],
            trial_length=data['trial_length'], optoAmp=args.optoAmp,
            seed=args.seed, reuse_wn=False, with_control=False)
        scaler = opto_model['scaler']
        if scaler is None:
            raise ValueError('Need a StandardScaler to map tanh rates to counts')
        counts_nt = tanh_rates_to_counts(
            sim['pred_opto'], scaler, opto_model['train_max'],
            poisson_mult=args.poisson_mult, seed=args.seed + 2)
        spikes_opto_bouts = counts_nt.reshape(
            n, int(sim['n_trials']), int(sim['trial_length']))
        opto_bins, stim_bin = opto_trial_bins(
            opto_ms=args.opto_ms, binsize_ms=20,
            stim_onset_ms=float(sim['stim_onset_s']) * 1000.0)
        if int(sim['stim_bin']) != int(stim_bin):
            raise ValueError('Simulator stim bin {} != export stim bin {}'.format(
                sim['stim_bin'], stim_bin))
        event_bin = stim_bin if args.input_event_bin is None else int(args.input_event_bin)
        spikes_opto, opto_behav, opto_start_bin = slice_opto_trials(
            spikes_opto_bouts, opto_bins)
        opto_starts = np.array([0], dtype=np.int32)
        cond_opto = np.array(['pseudo_opto'] * spikes_opto.shape[1])
        opto_extra = dict(
            source='pseudo_opto',
            init_tag=init_tag,
            opto_model=os.path.abspath(spec['path']),
            optoAmp=float(args.optoAmp),
            opto_target_population=sim['target_population'],
            opto_dur_s=float(sim['dur']),
            opto_stim_onset_s=float(sim['stim_onset_s']),
            stim_bin=int(event_bin),
            input_event_bin=int(event_bin),
            sampled_response_bin=int(stim_bin),
            input_is_binary_pulse=True,
            input_pulse_amplitude=1.0,
            physical_opto_amplitude=float(args.optoAmp),
            opto_window_ms=[0, int(args.opto_ms)],
            poisson_mult=float(args.poisson_mult),
        )
        if event_bin == stim_bin:
            opto_note = (
                'Pseudo-opto only ({init}, amp={amp}): deterministic I-cell pulse '
                'on {tgt}. One {ms} ms trial per bout from t=0; stim at '
                'timepoint {stim} ({onset} ms).'
                .format(init=init_tag, amp=args.optoAmp,
                        tgt=sim['target_population'], ms=int(args.opto_ms),
                        stim=event_bin, onset=int(sim['stim_onset_s'] * 1000))
            )
            merged_note = (
                'Climbing (100 ms) + pseudo-opto (200 ms, stim at bin {stim}). '
                'spikes is right-padded with NaN to T=10; use valid_mask or '
                'trial_lengths in the loss. trial_lists are native length. '
                'Do not treat padded bins as zero spikes.'
                .format(stim=event_bin)
            )
        else:
            opto_note = (
                'Pseudo-opto only ({init}, amp={amp}): deterministic I-cell pulse '
                'on {tgt}. One {ms} ms trial per bout from t=0; input event at '
                'timepoint {stim} (0-20 ms injection interval). First sampled '
                'rate response remains bin {resp} ({onset} ms).'
                .format(init=init_tag, amp=args.optoAmp,
                        tgt=sim['target_population'], ms=int(args.opto_ms),
                        stim=event_bin, resp=stim_bin,
                        onset=int(sim['stim_onset_s'] * 1000))
            )
            merged_note = (
                'Climbing (100 ms) + pseudo-opto (200 ms, input event at bin '
                '{stim}). spikes is right-padded with NaN to T=10; use '
                'valid_mask or trial_lengths in the loss. trial_lists are '
                'native length. Do not treat padded bins as zero spikes.'
                .format(stim=event_bin)
            )
        opto = pack_spike_export(
            stems['opto'], spikes_opto, opto_behav, opto_start_bin,
            opto_starts, opto_bins, condition=cond_opto, extra=opto_extra,
            note=opto_note,
            **pack_kw)
        write_pickle(spike_pickle_path(args.output_dir, stems['opto']), opto)

        pad_c, mask_c, len_c = pad_trials_right(spikes_tf, opto_bins)
        pad_o, mask_o, len_o = pad_trials_right(spikes_opto, opto_bins)
        spikes_m = np.concatenate([pad_c, pad_o], axis=1)
        mask_m = np.concatenate([mask_c, mask_o], axis=0)
        lengths_m = np.concatenate([len_c, len_o])
        behav_m = np.concatenate([behav_trial, opto_behav])
        start_m = np.concatenate([tf_start_bin, opto_start_bin])
        cond_m = np.concatenate([cond_climb, cond_opto])
        merged_extra = dict(opto_extra)
        merged_extra.update(
            source='climbing_plus_pseudo_opto',
            padded=True,
            pad_value=np.nan,
            n_climbing_windows=int(spikes_tf.shape[1]),
            n_opto_windows=int(spikes_opto.shape[1]),
            climbing_windows_per_bout=int(len(win_starts)),
            opto_windows_per_bout=1,
            spikes_climbing=spikes_tf,
            spikes_opto=spikes_opto,
        )
        merged = pack_spike_export(
            stems['merged'], spikes_m, behav_m, start_m, opto_starts, opto_bins,
            condition=cond_m, extra=merged_extra,
            trial_lengths=lengths_m, valid_mask=mask_m,
            note=merged_note,
            **pack_kw)
        write_pickle(spike_pickle_path(args.output_dir, stems['merged']), merged)


if __name__ == '__main__':
    main()
