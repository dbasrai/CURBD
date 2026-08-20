#!/usr/bin/env python
"""Screen HPs for Dale-constrained CURBD after the /max + radius failure.

Each config changes one (or one extra) knob relative to the previous idea.
Screening uses a trial subset so we can iterate; the best config is then
refit longer. Metrics: pVar (want >0.5), corr, saturation, spectral radius.
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
    'co10_01242024_climbing_pre20ms_post180ms_binsize20_'
    'reclassEI_inact0_35_mod0_25_aligned.pkl')

# Shared HPs copied from the failed run unless a config overrides them.
BASE = dict(
    g=1.5,
    g_across=1.5,
    g_loc=(-0.1, -0.1),
    tauRNN=0.05,
    ampInWN=0.001,
    P0=1.0,
    sparse_percent=60,
    adata_scale='max',
    target_radius=None,
    max_radius=None,
    project_every_update=True,
    dtFactor=5,
    seed=0,
    reset_every=None,
    reset_state='rate',
    align_error='prev',
    init_intra='dale',
    intra_dale='hard',
    intra_dale_ramp=(0.3, 0.8),
    project_intra_when='update',
    max_radius_after_frac=None,
    nRunTrain=12,
    nRunFree=1,
)

# Order is intentional: isolate scale, then radius, then timescale, then sparsity.
CONFIGS = [
    dict(
        name='A_baseline_max',
        why=(
            'Reproduce the failure on a trial subset. /max after z-score '
            'compresses data (std~0.03) while tanh lives in [-1,1]. If this '
            'is still ~ -800 pVar, the bug is not "too few full-dataset trials".'
        ),
        may_not_help='This is the control; it should still fail.',
    ),
    dict(
        name='B_scale_z3',
        adata_scale='z3',
        why=(
            'Divide z-scored rates by 3 then clip. Typical |z|~1 becomes 0.33, '
            '|z|~3 uses the full tanh range, outliers no longer set the scale. '
            'This is the most likely single fix.'
        ),
        may_not_help=(
            'Dale all-positive E columns still inflate spectral radius (~37). '
            'Units can saturate even with correctly scaled targets.'
        ),
    ),
    dict(
        name='C_scale_p99',
        adata_scale='p99',
        why=(
            'Data-driven alternative to z3: divide by the 99th |activity| '
            'percentile. Same idea as B if p99~3, more robust if the z-score '
            'width is not ~3.'
        ),
        may_not_help='Same radius/saturation issue as B if p99 is still small.',
    ),
    dict(
        name='D_z3_g08',
        adata_scale='z3',
        g=0.8,
        g_across=0.8,
        why=(
            'Lower g to offset Dale mean-field: folded-Gaussian E weights are '
            'all positive, so radius >> g. Smaller g reduces saturation.'
        ),
        may_not_help=(
            'g is a blunt knob; 0.8 may still be too large (or too small and '
            'underdamped / unable to generate the data).'
        ),
    ),
    dict(
        name='E_z3_radius12',
        adata_scale='z3',
        target_radius=1.2,
        why=(
            'After Dale init, rescale J so spectral radius is 1.2 (mildly '
            'chaotic, standard FORCE). Directly undoes the radius-37 problem '
            'without guessing g.'
        ),
        may_not_help=(
            'RLS + Dale clipping can grow radius again during training. Init '
            'rescale does not constrain the trained J.'
        ),
    ),
    dict(
        name='F_z3_radius12_tau02',
        adata_scale='z3',
        target_radius=1.2,
        tauRNN=0.2,
        why=(
            'Original Bio sweep used tau/dtData=10 with 5 ms bins. Here '
            'dtData=20 ms, so tauRNN=0.2 s restores that ratio. Helps the '
            'RNN integrate on the data timescale.'
        ),
        may_not_help=(
            'If scale+radius already suffice, tau is secondary. Too-slow tau '
            'can smear 20 ms transients.'
        ),
    ),
    dict(
        name='G_z3_radius12_tau02_sparse0',
        adata_scale='z3',
        target_radius=1.2,
        tauRNN=0.2,
        sparse_percent=0,
        why=(
            'Do not zero 30-60% of cross-region E columns while the network '
            'is still learning. Sparsity on a saturated/unfit J removes the '
            'wrong weights. Fit first, sparsify later.'
        ),
        may_not_help=(
            'If E/F already fit, extra density just adds degeneracy. Cross '
            'weights may need sparsity for a Bio-like J, but not for pVar.'
        ),
    ),
    dict(
        name='H_z3_radius12_tau02_sparse0_P05',
        adata_scale='z3',
        target_radius=1.2,
        tauRNN=0.2,
        sparse_percent=0,
        P0=0.5,
        why=(
            'Halve FORCE gain. Dale projection after every RLS step can make '
            'P0=1 overshoot (clip, then compensate, then clip).'
        ),
        may_not_help='If G is already stable, smaller P0 only slows learning.',
    ),
    # Wave 2: radius grows from 1.2 to 20-50 during RLS. Cap it.
    dict(
        name='I_z3_maxrad12',
        adata_scale='z3',
        target_radius=1.2,
        max_radius=1.2,
        sparse_percent=0,
        why=(
            'Wave 1: z3 fixed scale (pVar -900 to -13) but J radius still grew '
            'to ~20-50 and units stayed saturated. Rescale J at trial resets '
            'and epoch ends so radius cannot exceed 1.2.'
        ),
        may_not_help=(
            'Capping radius fights FORCE: if the unconstrained J needs large '
            'gain to match data, the fit will stall. Also eig-rescale changes '
            'Dale column magnitudes uniformly, which may not be the right '
            'error direction.'
        ),
    ),
    dict(
        name='J_z3_maxrad15',
        adata_scale='z3',
        target_radius=1.2,
        max_radius=1.5,
        sparse_percent=0,
        why=(
            'Allow slightly more gain than I (radius up to 1.5) so FORCE can '
            'grow weights a bit without saturating at 20+.'
        ),
        may_not_help='1.5 may still saturate tanh, or still be too tight for FORCE.',
    ),
    dict(
        name='K_z3_maxrad12_tau02',
        adata_scale='z3',
        target_radius=1.2,
        max_radius=1.2,
        tauRNN=0.2,
        sparse_percent=0,
        why='Radius cap plus Bio-matched tau/dtData=10.',
        may_not_help='Tau is secondary if radius cap is the real fix (or blocker).',
    ),
    dict(
        name='L_z3_maxrad12_P01',
        adata_scale='z3',
        target_radius=1.2,
        max_radius=1.2,
        sparse_percent=0,
        P0=0.1,
        why=(
            'Smaller FORCE steps so J moves less between radius rescales. '
            'P0=0.5 in wave 1 still exploded radius within an epoch.'
        ),
        may_not_help='May learn too slowly in 12 runs; would need more nRunTrain.',
    ),
    dict(
        name='M_z3_maxrad12_tau02_P01',
        adata_scale='z3',
        target_radius=1.2,
        max_radius=1.2,
        tauRNN=0.2,
        sparse_percent=0,
        P0=0.1,
        why='Combine radius cap, slower tau, and smaller FORCE gain.',
        may_not_help='Three knobs at once; if it works we should ablate later.',
    ),
    dict(
        name='N_K_dale_at_resets',
        adata_scale='z3',
        target_radius=1.2,
        max_radius=1.2,
        tauRNN=0.2,
        sparse_percent=0,
        project_every_update=False,
        why=(
            'K plateaued at pVar~-0.6 with 0% saturation. Projecting Dale '
            'after every RLS step may cancel the FORCE update. Project only '
            'at trial resets / epoch ends (plus radius cap).'
        ),
        may_not_help=(
            'Weights can violate Dale between resets. If that is what lets '
            'pVar rise, the fit is not a Dale network during learning.'
        ),
    ),
    # Wave 5: initialization draws, 20ms vs 10ms timing, reset schedule.
    dict(
        name='P_seed1',
        adata_scale='z3', target_radius=1.2, max_radius=1.2, tauRNN=0.2,
        sparse_percent=0, seed=1,
        why=(
            'Same HPs as K (best wave 2) but a different random J0. Dale '
            'folded-Gaussian init is high-variance; one seed can sit in a '
            'bad basin even after radius rescaling.'
        ),
        may_not_help='If the plateau is structural (Dale+radius cap), seeds will cluster.',
    ),
    dict(
        name='P_seed2',
        adata_scale='z3', target_radius=1.2, max_radius=1.2, tauRNN=0.2,
        sparse_percent=0, seed=2,
        why='Second init draw of K.',
        may_not_help='Same as P_seed1.',
    ),
    dict(
        name='P_seed7',
        adata_scale='z3', target_radius=1.2, max_radius=1.2, tauRNN=0.2,
        sparse_percent=0, seed=7,
        why='Third init draw of K.',
        may_not_help='Same as P_seed1.',
    ),
    dict(
        name='Q_dtFactor10',
        adata_scale='z3', target_radius=1.2, max_radius=1.2, tauRNN=0.2,
        sparse_percent=0, dtFactor=10,
        why=(
            '20 ms bins with dtFactor=5 give dtRNN=4 ms. Old 10 ms bins with '
            'dtFactor=5 had dtRNN=2 ms. dtFactor=10 restores that Euler step '
            'so FORCE updates integrate the ODE at the previous resolution.'
        ),
        may_not_help=(
            'RLS still fires once per 20 ms data bin, so this only changes '
            'interpolation. If Euler was already stable, pVar will not move.'
        ),
    ),
    dict(
        name='R_tau01',
        adata_scale='z3', target_radius=1.2, max_radius=1.2, tauRNN=0.1,
        sparse_percent=0,
        why=(
            '10 ms data + tauRNN=0.05 had tau/dtData=5. 20 ms + tau=0.05 is '
            'only 2.5 bins. tauRNN=0.1 restores tau/dtData=5. K used 0.2 '
            '(ratio 10, like 5 ms Bio). This tests the 10 ms-matched tau.'
        ),
        may_not_help='K already beat tau=0.05; 0.1 may sit between 0.05 and 0.2.',
    ),
    dict(
        name='S_dtF10_tau01',
        adata_scale='z3', target_radius=1.2, max_radius=1.2, tauRNN=0.1,
        sparse_percent=0, dtFactor=10,
        why='Combine 2 ms Euler step with 10 ms-matched tau/dtData=5.',
        may_not_help='Two timing knobs; if either is enough the other is noise.',
    ),
    dict(
        name='T_reset100',
        adata_scale='z3', target_radius=1.2, max_radius=1.2, tauRNN=0.2,
        sparse_percent=0, reset_every=100,
        why=(
            'Bio sweep teacher-forced every 100 RNN steps, not just trial '
            'starts. At dtFactor=5 a 41-bin trial is 205 steps, so this adds '
            'a mid-trial reset (~400 ms), same data-time spacing as 100 steps '
            'on 5 ms/dtFactor=5.'
        ),
        may_not_help=(
            'More resets inflate pVar by copying data into H without the '
            'RNN actually generating it. Autonomous nRunFree pVar may stay bad.'
        ),
    ),
    dict(
        name='T_reset50',
        adata_scale='z3', target_radius=1.2, max_radius=1.2, tauRNN=0.2,
        sparse_percent=0, reset_every=50,
        why=(
            '10 ms bins + dtFactor=5 + num_reset=100 was a reset every 200 ms. '
            '20 ms bins + dtFactor=5 need reset_every=50 to match that '
            'wall-clock interval (T_reset100 is 400 ms, twice as sparse).'
        ),
        may_not_help='Same teacher-force caveat as T_reset100; pVar can look good while free-run does not.',
    ),
    dict(
        name='U_reset_current',
        adata_scale='z3', target_radius=1.2, max_radius=1.2, tauRNN=0.2,
        sparse_percent=0, reset_state='current',
        why=(
            'Legacy reset sets H=Adata then RNN=tanh(H), so the network does '
            'not start on the data. reset_state=current uses H=arctanh(Adata) '
            'so rates match the target at every reset.'
        ),
        may_not_help='z3 rates are ~0.3 so tanh(H) was already close; small effect.',
    ),
    dict(
        name='V_align_current',
        adata_scale='z3', target_radius=1.2, max_radius=1.2, tauRNN=0.2,
        sparse_percent=0, align_error='current',
        why=(
            'FORCE currently compares RNN at t=dtData to Adata[:,0] (one-bin '
            'lag). At 20 ms that lag is 2x a 10 ms dataset. Align error to '
            'the data sample at the current RNN time.'
        ),
        may_not_help='A one-bin lag is in the original CURBD code; it still fit 5–10 ms data.',
    ),
    dict(
        name='W_gloc0',
        adata_scale='z3', target_radius=1.2, max_radius=1.2, tauRNN=0.2,
        sparse_percent=0, g_loc=(0.0, 0.0),
        why=(
            'g_loc=-0.1 shifts local E toward 0 (and clips ~5%) and local I '
            'more negative before radius rescale. g_loc=0 keeps folded-'
            'Gaussian E/I symmetric in magnitude, changing J0 shape not just scale.'
        ),
        may_not_help='Radius rescale may wash out a 0.1 shift.',
    ),
    dict(
        name='X_gacross03',
        adata_scale='z3', target_radius=1.2, max_radius=1.2, tauRNN=0.2,
        sparse_percent=0, g_across=0.3,
        why=(
            'Cross-region E init is (g_across/g)*|N(0,1)|. Lower g_across '
            'makes J0 more local before the radius rescale, i.e. a different '
            'init direction not just a different seed.'
        ),
        may_not_help='Uniform radius rescale can re-amplify the remaining local block.',
    ),
    dict(
        name='Y_radius08',
        adata_scale='z3', target_radius=0.8, max_radius=0.8, tauRNN=0.2,
        sparse_percent=0,
        why=(
            'Damped init (radius 0.8). K used 1.2 (mildly chaotic). If Dale '
            'columns still push the network too hard between rescales, a '
            'smaller init/cap can keep rates in the linear tanh region.'
        ),
        may_not_help='Too-small radius cannot generate the data; pVar may stall lower.',
    ),
    dict(
        name='Z_timing_reset_combo',
        adata_scale='z3', target_radius=1.2, max_radius=1.2, tauRNN=0.1,
        sparse_percent=0, dtFactor=10, reset_every=100,
        reset_state='current', align_error='current',
        why=(
            'Stack 10 ms-matched timing (dtRNN=2 ms, tau/dt=5), Bio-style '
            'intra-trial resets, arctanh reset, and current-bin FORCE error.'
        ),
        may_not_help='Confounded combo; use only if it clearly beats K, then ablate.',
    ),
    dict(
        name='AA_g08_radius12',
        adata_scale='z3', target_radius=1.2, max_radius=1.2, tauRNN=0.2,
        sparse_percent=0, g=0.8, g_across=0.8,
        why=(
            'Radius rescale fixes ||J|| but not shape. Smaller g before '
            'rescale makes the g_loc=-0.1 offset a larger fraction of local '
            'weights (more E mass near 0, heavier I after clip).'
        ),
        may_not_help='If FORCE overwrites J0 in a few epochs, init shape will not matter.',
    ),
    dict(
        name='AB_radius09',
        adata_scale='z3', target_radius=0.9, max_radius=0.9, tauRNN=0.2,
        sparse_percent=0,
        why='Interpolate Y (0.8, underpowered) and K (1.2, overshoots).',
        may_not_help='May sit between the two without crossing pVar=0.',
    ),
    dict(
        name='AC_radius10',
        adata_scale='z3', target_radius=1.0, max_radius=1.0, tauRNN=0.2,
        sparse_percent=0,
        why='Unit-radius cap: enough gain to match data std if 0.8 was the limiter.',
        may_not_help='Dale mean-field may still need <1 to stay linear.',
    ),
    dict(
        name='AD_radius08_reset50',
        adata_scale='z3', target_radius=0.8, max_radius=0.8, tauRNN=0.2,
        sparse_percent=0, reset_every=50,
        why='Best honest init (Y) plus 10 ms-matched 200 ms teacher-force (T_reset50).',
        may_not_help='T_reset50 already got pVar from resets, not FORCE; damping may not add a real fit.',
    ),
    dict(
        name='AE_radius08_dale_resets',
        adata_scale='z3', target_radius=0.8, max_radius=0.8, tauRNN=0.2,
        sparse_percent=0, project_every_update=False,
        why=(
            'Dale clip after every RLS step can cancel FORCE. Project only at '
            'trial resets, with the damped radius that stopped saturation.'
        ),
        may_not_help='If pVar rises only while Dale is violated, J is not a Dale GT.',
    ),
    # Stage A: Bio-style signed intra, no intra Dale, no radius cap.
    dict(
        name='SA_bio_intra',
        adata_scale='z3', tauRNN=0.2, sparse_percent=0,
        target_radius=None, max_radius=None,
        init_intra='signed', intra_dale='off',
        project_every_update=True, project_intra_when='epoch',
        why=(
            'Diagnostic: same 20 ms z3 data and K timescales, but Bio signed '
            'intra init, no intra Dale, no radius cap. Inter E-only stays hard. '
            'If pVar never reaches 0.5, the bottleneck is data/FORCE not Dale.'
        ),
        may_not_help='Signed intra may still need Bio-style mid-trial resets.',
    ),
    dict(
        name='SA_bio_reset50',
        adata_scale='z3', tauRNN=0.2, sparse_percent=0, reset_every=50,
        target_radius=None, max_radius=None,
        init_intra='signed', intra_dale='off',
        project_every_update=True, project_intra_when='epoch',
        why='Stage A plus 200 ms teacher-force (10 ms Bio num_reset match).',
        may_not_help='pVar can look good from resets without learned dynamics.',
    ),
    dict(
        name='SA_bio_sparse60',
        adata_scale='z3', tauRNN=0.2, sparse_percent=60,
        target_radius=None, max_radius=None,
        init_intra='signed', intra_dale='off',
        project_every_update=True, project_intra_when='epoch',
        why='Stage A plus Bio-style 60% weak inter-region sparsify.',
        may_not_help='Sparsity during an unconstrained intra fit may not be the limiter.',
    ),
    dict(
        name='SA_bio_rad12',
        adata_scale='z3', tauRNN=0.2, sparse_percent=0,
        target_radius=1.2, max_radius=1.2,
        init_intra='signed', intra_dale='off',
        project_every_update=True, project_intra_when='epoch',
        why=(
            'Stage A exploded to rho~250 without a cap. Signed intra plus '
            'radius 1.2 tests whether FORCE can fit when gain is stable but '
            'intra Dale is still off.'
        ),
        may_not_help='Cap may again underpower; then try P0 or 0.8.',
    ),
    dict(
        name='SA_bio_P01',
        adata_scale='z3', tauRNN=0.2, sparse_percent=0, P0=0.1,
        target_radius=None, max_radius=None,
        init_intra='signed', intra_dale='off',
        project_every_update=True, project_intra_when='epoch',
        why='Smaller FORCE steps instead of a radius cap. Isolates RLS gain.',
        may_not_help='P0=0.1 may still explode over 25 runs, or learn too slowly.',
    ),
    dict(
        name='SA_bio_rad12_reset50',
        adata_scale='z3', tauRNN=0.2, sparse_percent=0, reset_every=50,
        target_radius=1.2, max_radius=1.2,
        init_intra='signed', intra_dale='off',
        project_every_update=True, project_intra_when='epoch',
        why='Stable-gain Stage A plus 200 ms teacher-force.',
        may_not_help='Teacher-force can inflate pVar without a real fit.',
    ),
    dict(
        name='SA_bio_rad08',
        adata_scale='z3', tauRNN=0.2, sparse_percent=0,
        target_radius=0.8, max_radius=0.8,
        init_intra='signed', intra_dale='off',
        project_every_update=True, project_intra_when='epoch',
        why='Y-style radius 0.8 with signed intra (no Dale).',
        may_not_help='Underpowered like Y if amplitude is the limiter, not Dale.',
    ),
    dict(
        name='SA_bio_rad12_P01',
        adata_scale='z3', tauRNN=0.2, sparse_percent=0, P0=0.1,
        target_radius=1.2, max_radius=1.2,
        init_intra='signed', intra_dale='off',
        project_every_update=True, project_intra_when='epoch',
        why='P0=0.1 peaked at pVar~0.06 before exploding; radius cap should keep that basin.',
        may_not_help='May plateau below 0.5 like other capped runs.',
    ),
    # Stage B: anneal intra Dale onto a Bio-like basin.
    dict(
        name='SB_anneal_epoch',
        adata_scale='z3', tauRNN=0.2, sparse_percent=0,
        target_radius=1.2, max_radius=1.2,
        init_intra='signed', intra_dale='anneal', intra_dale_ramp=(0.3, 0.8),
        project_intra_when='epoch', project_every_update=True,
        why=(
            'Linear intra Dale anneal, project intra only at epoch end. '
            'Slowest application of the constraint.'
        ),
        may_not_help='Illegal mass can grow all epoch and a single leak may not fold onto Dale.',
    ),
    dict(
        name='SB_anneal_reset',
        adata_scale='z3', tauRNN=0.2, sparse_percent=0,
        target_radius=1.2, max_radius=1.2,
        init_intra='signed', intra_dale='anneal', intra_dale_ramp=(0.3, 0.8),
        project_intra_when='reset', project_every_update=True,
        why='Same anneal, leak intra Dale at trial resets (Bio-like frequency).',
        may_not_help='Still only a few projections per trial.',
    ),
    dict(
        name='SB_anneal_update',
        adata_scale='z3', tauRNN=0.2, sparse_percent=0,
        target_radius=1.2, max_radius=1.2,
        init_intra='signed', intra_dale='anneal', intra_dale_ramp=(0.3, 0.8),
        project_intra_when='update', project_every_update=True,
        why='Same anneal, leak intra after every RLS step once alpha>0.',
        may_not_help='May still fight FORCE once alpha is large, just later.',
    ),
    dict(
        name='SB_anneal_late',
        adata_scale='z3', tauRNN=0.2, sparse_percent=0,
        target_radius=1.2, max_radius=1.2,
        init_intra='signed', intra_dale='anneal', intra_dale_ramp=(0.6, 0.95),
        project_intra_when='epoch', project_every_update=True,
        why='Hold alpha=0 until 60% of runs, then ramp. Longer Bio-like basin.',
        may_not_help='A late hard clip can still destroy pVar in a few epochs.',
    ),
    dict(
        name='SB_anneal_epoch_rad',
        adata_scale='z3', tauRNN=0.2, sparse_percent=0,
        target_radius=1.2, max_radius=1.2, max_radius_after_frac=0.66,
        init_intra='signed', intra_dale='anneal', intra_dale_ramp=(0.3, 0.8),
        project_intra_when='epoch', project_every_update=True,
        why='Best-guess Stage B plus radius cap only in the last third, after Dale appears.',
        may_not_help='1.2 cap may still underpower a newly Dale-folded J.',
    ),
    dict(
        name='SB_anneal_epoch_P01',
        adata_scale='z3', tauRNN=0.2, sparse_percent=0, P0=0.1,
        target_radius=1.2, max_radius=1.2,
        init_intra='signed', intra_dale='anneal', intra_dale_ramp=(0.3, 0.8),
        project_intra_when='epoch', project_every_update=True,
        why='Anneal intra Dale on the best honest Stage A basin (P0=0.1, radius 1.2).',
        may_not_help='That basin was already ~0 pVar; hardening Dale may drop it.',
    ),
]


def merge_cfg(cfg):
    out = dict(BASE)
    for k, v in cfg.items():
        if k not in ('name', 'why', 'may_not_help'):
            out[k] = v
    out['name'] = cfg['name']
    out['why'] = cfg['why']
    out['may_not_help'] = cfg['may_not_help']
    return out


def run_one(data, cfg, outdir, seed=None):
    if seed is None:
        seed = int(cfg.get('seed', 0))
    npr.seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    dtFactor = int(cfg.get('dtFactor', 5))
    resetPoints = curbd.make_reset_points(
        data['n_trials'], data['trial_length'], dtFactor,
        reset_every=cfg.get('reset_every'))
    print('\n' + '=' * 72)
    print('CONFIG', cfg['name'])
    print('WHY:', cfg['why'])
    print('MAY NOT HELP:', cfg['may_not_help'])
    print('HPs:', {k: cfg.get(k) for k in (
        'adata_scale', 'g', 'g_across', 'g_loc', 'target_radius', 'max_radius',
        'max_radius_after_frac', 'tauRNN', 'dtFactor', 'reset_every',
        'reset_state', 'align_error', 'sparse_percent', 'P0', 'seed',
        'nRunTrain', 'init_intra', 'intra_dale', 'intra_dale_ramp',
        'project_intra_when')})
    model = curbd.trainEIBioMultiRegionRNN(
        data['z_activity'],
        dtData=data['dtData'],
        dtFactor=dtFactor,
        tauRNN=cfg['tauRNN'],
        ampInWN=cfg['ampInWN'],
        regions=data['regions'],
        populations=data['populations'],
        ei_sign=data['ei_sign'],
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
        adata_scale=cfg['adata_scale'],
        target_radius=cfg['target_radius'],
        max_radius=cfg.get('max_radius'),
        project_every_update=cfg.get('project_every_update', True),
        reset_state=cfg.get('reset_state', 'rate'),
        align_error=cfg.get('align_error', 'prev'),
        init_intra=cfg.get('init_intra', 'dale'),
        intra_dale=cfg.get('intra_dale', 'hard'),
        intra_dale_ramp=cfg.get('intra_dale_ramp', (0.3, 0.8)),
        project_intra_when=cfg.get('project_intra_when', 'update'),
        max_radius_after_frac=cfg.get('max_radius_after_frac'),
    )
    model['scaler'] = data['scaler']
    model['dataset_name'] = data['name']
    model['hp_name'] = cfg['name']
    metrics = curbd.summarize_ei_fit(model)
    dale = curbd.check_dale_constraints(model)
    elapsed = time.time() - t0
    rho_final = curbd._spectral_radius(model['J'])
    row = dict(
        name=cfg['name'],
        adata_scale=cfg['adata_scale'],
        g=cfg['g'],
        target_radius=cfg['target_radius'],
        max_radius=cfg.get('max_radius'),
        project_every_update=cfg.get('project_every_update', True),
        tauRNN=cfg['tauRNN'],
        sparse_percent=cfg['sparse_percent'],
        P0=cfg['P0'],
        nRunTrain=cfg['nRunTrain'],
        dtFactor=dtFactor,
        seed=seed,
        reset_every=cfg.get('reset_every'),
        reset_state=cfg.get('reset_state', 'rate'),
        align_error=cfg.get('align_error', 'prev'),
        g_loc=str(cfg['g_loc']),
        g_across=cfg['g_across'],
        init_intra=cfg.get('init_intra', 'dale'),
        intra_dale=cfg.get('intra_dale', 'hard'),
        intra_dale_ramp=str(cfg.get('intra_dale_ramp')),
        project_intra_when=cfg.get('project_intra_when', 'update'),
        max_radius_after_frac=cfg.get('max_radius_after_frac'),
        status=metrics['status'],
        pvar0=float(metrics['pVars'][0]) if len(metrics['pVars']) else np.nan,
        pvar_final=metrics['pvar_final'],
        chi2_final=metrics['chi2_final'],
        corr=metrics['corr'],
        r2_median=metrics['r2_median'],
        sat_model=metrics['sat_model'],
        Adata_std=metrics['Adata_std'],
        rho_init=model['params'].get('rho_init'),
        rho_final=rho_final,
        dale_ok=dale['ok']['all'],
        dale_intra_ok=dale['ok']['intra_e_nonneg'] and dale['ok']['intra_i_nonpos'],
        dale_inter_ok=dale['ok']['inter_e_nonneg'] and dale['ok']['inter_i_zero'],
        dale_alpha_final=metrics.get('dale_alpha_final'),
        illegal_intra_final=metrics.get('illegal_intra_final'),
        elapsed_s=elapsed,
    )
    print('RESULT {name}: status={status} pVar {pvar0:.3f}->{pvar_final:.3f} '
          'corr={corr:.3f} sat={sat_model:.1%} rho {rho_init:.2f}->{rho_final:.2f} '
          'dale_intra={dale_intra_ok} illegal={illegal_intra_final:.3g} '
          '{elapsed_s:.0f}s'.format(**row))
    run_dir = os.path.join(outdir, cfg['name'])
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'model.pickle'), 'wb') as f:
        pickle.dump({'model': model, 'ground_truth': None, 'metrics': row}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    curbd.plot_convergence(model)[0].savefig(
        os.path.join(run_dir, 'convergence.png'), dpi=120, bbox_inches='tight')
    curbd.plot_rate_match(model, n_trials=6)[0].savefig(
        os.path.join(run_dir, 'rates.png'), dpi=120, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close('all')
    return row, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--output-dir', default=os.path.join(ROOT, 'outputs', 'hp_sweep'))
    parser.add_argument('--max-trials', type=int, default=40,
                        help='Subset of trials for screening (speed)')
    parser.add_argument('--nRunTrain', type=int, default=12)
    parser.add_argument('--only', default=None,
                        help='Comma-separated config names to run')
    parser.add_argument('--refit-best', action='store_true',
                        help='After screening, refit the best config on more trials')
    parser.add_argument('--refit-trials', type=int, default=80)
    parser.add_argument('--refit-runs', type=int, default=40)
    args = parser.parse_args()

    only = None
    if args.only:
        only = set(x.strip() for x in args.only.split(','))

    data = curbd.load_ei_dataset(
        args.dataset, dtFactor=5, smooth_sigma=1.5, zscore=True,
        max_trials=args.max_trials)
    print('Screening data {}  N={} T={} trials={}'.format(
        data['name'], data['z_activity'].shape[0], data['z_activity'].shape[1],
        data['n_trials']))

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'sweep_log.csv')
    rows = []
    best = None
    best_model = None
    for raw_cfg in CONFIGS:
        if only and raw_cfg['name'] not in only:
            continue
        cfg = merge_cfg(raw_cfg)
        cfg['nRunTrain'] = args.nRunTrain
        row, model = run_one(data, cfg, args.output_dir)
        rows.append(row)
        score = row['pvar_final']
        if best is None or score > best['pvar_final']:
            best = row
            best_model = model

    fieldnames = list(rows[0].keys()) if rows else []
    with open(log_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('\nWrote', log_path)
    print('\nSCREENING SUMMARY (sorted by pVar):')
    for r in sorted(rows, key=lambda x: x['pvar_final'], reverse=True):
        print('  {name:40s}  pVar={pvar_final:8.3f}  corr={corr:6.3f}  '
              'sat={sat_model:6.1%}  status={status}'.format(**r))

    if args.refit_best and best is not None:
        print('\nREFIT BEST:', best['name'], 'pVar', best['pvar_final'])
        raw = [c for c in CONFIGS if c['name'] == best['name']][0]
        cfg = merge_cfg(raw)
        cfg['nRunTrain'] = args.refit_runs
        data_full = curbd.load_ei_dataset(
            args.dataset, dtFactor=5, smooth_sigma=1.5, zscore=True,
            max_trials=args.refit_trials)
        row, model = run_one(data_full, cfg, os.path.join(args.output_dir, 'refit'))
        with open(os.path.join(args.output_dir, 'refit', 'best_row.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        curbd.diagnose_ei_model(
            model, outdir=os.path.join(args.output_dir, 'refit'),
            prefix='best_{}'.format(best['name']), show=False)


if __name__ == '__main__':
    main()
