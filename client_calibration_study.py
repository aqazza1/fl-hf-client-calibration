"""Client-level calibration of a federated HF mortality model under temporal shift.

Runs locally on credentialed MIMIC-IV v3.1. Reads the audited feature export and
the source tables, keeps identifiers and dates in memory only, and writes
aggregate results. No patient-level rows, predictions or weights are written.

Usage (from the project root that holds build_cohort.py, q1_experiments.py,
run_experiments.py and local_output/):

    python client_calibration_study.py --root "<mimic-iv-3.1 folder>" --out local_output/client_calibration
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from build_cohort import LABEL, extract_index, read
from q1_experiments import fit
from run_experiments import features

import calib_core as cc

THRESHOLDS = (0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20)
PRIMARY = (0.05, 0.10)
SEED = 20261106
COHORT_HASH = '8ec032ce61535a45acb927ef0a773c37cf1fec39acce00b0dda9c408bd2c992d'
MIN_CELL = 30
CURVE_SIZES = (25, 50, 100, 200, 400, 800)
CURVE_REPS = 200
LR_MIN_EVENTS = 10

SERVICE_GROUP = {
    'CMED': 'Cardiac medicine',
    'MED': 'General medicine',
    'CSURG': 'Cardiovascular and thoracic surgery',
    'VSURG': 'Cardiovascular and thoracic surgery',
    'TSURG': 'Cardiovascular and thoracic surgery',
}
OTHER = 'Other services'

UPDATES = ('FedAvg global', 'Federated CITL', 'Federated logistic recalibration',
           'Local CITL', 'Local logistic recalibration',
           'Personalized FL', 'Personalized FL + local CITL',
           'Pooled', 'Local-only')
REFERENCE = 'FedAvg global'


# ------------------------------------------------------------------ cohort

def temporal_split(root, cohort, validation_year, test_year):
    """Identical to final_validation.temporal_split (audited design)."""
    anchors = read(root, 'hosp/patients', ['subject_id', 'anchor_year_group'])
    groups = cohort.subject_id.map(anchors.set_index('subject_id').anchor_year_group)
    bounds = groups.str.extract(r'^(\d{4}) - (\d{4})$').astype(int)
    if not bounds[1].sub(bounds[0]).eq(2).all():
        raise ValueError('Unexpected anchor-year group width')
    admission_offset = cohort.admittime.dt.year - cohort.anchor_year
    discharge_offset = cohort.dischtime.dt.year - cohort.anchor_year
    earliest_admission = bounds[0] + admission_offset
    latest_discharge = bounds[1] + discharge_offset
    split = pd.Series('excluded', index=cohort.index)
    split.loc[latest_discharge.lt(validation_year)] = 'train'
    split.loc[earliest_admission.ge(validation_year) & latest_discharge.lt(test_year)] = 'validation'
    split.loc[earliest_admission.ge(test_year)] = 'test'
    return split


def landmark_service(root, index):
    """Service in effect at the 24-hour landmark (last transfer at or before it)."""
    if 'hadm_id' not in index.columns:
        admissions = read(root, 'hosp/admissions', ['subject_id', 'hadm_id', 'admittime'])
        admissions['admittime'] = pd.to_datetime(admissions.admittime)
        index = index.merge(admissions, on=['subject_id', 'admittime'], how='left', validate='one_to_one')
        if index.hadm_id.isna().any():
            raise ValueError('Could not link every index admission to hadm_id')
    services = read(root, 'hosp/services', ['hadm_id', 'transfertime', 'curr_service'])
    services['transfertime'] = pd.to_datetime(services.transfertime)
    keep = services.merge(index[['hadm_id', 'admittime']], on='hadm_id')
    keep = keep.loc[keep.transfertime <= keep.admittime + pd.Timedelta(hours=24)]
    last = keep.sort_values('transfertime').groupby('hadm_id').curr_service.last()
    return index.hadm_id.map(last).fillna('UNRECORDED').to_numpy()


def adequate(groups, labels, split):
    """True for groups with >= MIN_CELL events and non-events in every era."""
    table = pd.DataFrame(dict(g=groups, y=labels, s=split))
    table = table.loc[table.s.isin(['train', 'validation', 'test'])]
    counts = table.groupby(['g', 's']).y.agg(['sum', 'size'])
    ok = {}
    for g in np.unique(groups):
        rows = [counts.loc[(g, s)] if (g, s) in counts.index else None for s in ('train', 'validation', 'test')]
        ok[g] = all(r is not None and r['sum'] >= MIN_CELL and r['size'] - r['sum'] >= MIN_CELL for r in rows)
    return ok


def merge_small(groups, labels, split, fallback):
    """Prespecified merging: failing groups go to fallback(groups) until all pass."""
    groups = np.asarray(groups, dtype=object).copy()
    log = []
    for _ in range(20):
        ok = adequate(groups, labels, split)
        failing = sorted(g for g, good in ok.items() if not good)
        if not failing:
            return groups, log
        # Specific groups are merged before the residual group, smallest first.
        source = min(failing, key=lambda g: (g == OTHER, (groups == g).sum()))
        target = fallback(groups, source)
        if target == source:
            raise ValueError(f'No merge target for {source}')
        log.append(dict(merged=source, into=target))
        groups[groups == source] = target
    raise ValueError('Group merging did not terminate')


def service_fallback(groups, source):
    if source != OTHER and OTHER in set(groups):
        return OTHER
    return 'General medicine'


def largest_fallback(groups, source):
    values, counts = np.unique(groups[groups != source], return_counts=True)
    return values[np.argmax(counts)]


# ------------------------------------------------------------------ updating maps

def update_maps(eta_val, y_val, c_val, eta_pfl_val, clients):
    """Fit every updating map on validation-era data. Returns params and fallbacks."""
    params, fallback = {}, {}
    parts = [(eta_val[c_val == k], y_val[c_val == k]) for k in clients]
    a, _ = cc.federated_newton(parts, 1)
    params['Federated CITL'] = None if a is None else float(a[0])
    ab, _ = cc.federated_newton(parts, 2)
    params['Federated logistic recalibration'] = None if ab is None else tuple(map(float, ab))
    local_a, local_ab, pfl_a = {}, {}, {}
    for k, (eta_k, y_k) in zip(clients, parts):
        a_k = cc.fit_offset(eta_k, y_k)
        fallback[f'Local CITL|{k}'] = a_k is None
        local_a[k] = 0.0 if a_k is None else a_k
        ab_k = cc.fit_logistic_recal(eta_k, y_k) if y_k.sum() >= LR_MIN_EVENTS else None
        fallback[f'Local logistic recalibration|{k}'] = ab_k is None
        local_ab[k] = (local_a[k], 1.0) if ab_k is None else ab_k
        mask = c_val == k
        p_k = cc.fit_offset(eta_pfl_val[mask], y_val[mask])
        pfl_a[k] = 0.0 if p_k is None else p_k
    params['Local CITL'] = local_a
    params['Local logistic recalibration'] = local_ab
    params['Personalized FL + local CITL'] = pfl_a
    return params, fallback


def predict(method, params, eta, eta_pfl, eta_pool, eta_local, c):
    if method == 'FedAvg global':
        z = eta
    elif method == 'Federated CITL':
        z = eta + (params[method] or 0.0)
    elif method == 'Federated logistic recalibration':
        a, b = params[method] or (0.0, 1.0)
        z = a + b * eta
    elif method == 'Local CITL':
        z = eta + np.vectorize(params[method].get)(c)
    elif method == 'Local logistic recalibration':
        a = np.array([params[method][k][0] for k in c])
        b = np.array([params[method][k][1] for k in c])
        z = a + b * eta
    elif method == 'Personalized FL':
        z = eta_pfl
    elif method == 'Personalized FL + local CITL':
        z = eta_pfl + np.vectorize(params[method].get)(c)
    elif method == 'Pooled':
        z = eta_pool
    elif method == 'Local-only':
        z = eta_local
    else:
        raise KeyError(method)
    return cc.sigmoid(z)


def bins(y, p):
    out = []
    for low, high in zip((0, 0.02, 0.05, 0.10, 0.20, 0.40), (0.02, 0.05, 0.10, 0.20, 0.40, 1.0)):
        m = (p >= low) & (p < high)
        if m.sum() >= MIN_CELL:
            out.append(dict(mean_risk=float(p[m].mean()), event_rate=float(y[m].mean()), patients=int(m.sum())))
    return out


def evaluate_all(preds, y, c, clients, light=False):
    """Metrics overall and per client for every method."""
    result = {}
    for method, p in preds.items():
        row = dict(overall=cc.metrics(y, p, THRESHOLDS), clients={})
        for k in clients:
            m = c == k
            row['clients'][k] = cc.metrics(y[m], p[m], THRESHOLDS)
            if not light:
                row['clients'][k]['bins'] = bins(y[m], p[m])
        if not light:
            row['overall']['bins'] = bins(y, p)
        citl = [row['clients'][k]['citl'] for k in clients]
        ici = [row['clients'][k]['ici'] for k in clients]
        n = np.array([row['clients'][k]['n'] for k in clients], float)
        row['summary'] = dict(
            max_abs_citl=None if None in citl else float(np.max(np.abs(citl))),
            sd_citl=None if None in citl else float(np.std(citl)),
            weighted_ici=None if None in ici else float(np.sum(n * np.array(ici)) / n.sum()),
            max_ici=None if None in ici else float(np.max(ici)))
        result[method] = row
    return result


def flatten(evaluation):
    """Scalar view used to build bootstrap distributions."""
    flat = {}
    for method, row in evaluation.items():
        for key, value in row['summary'].items():
            flat[(method, 'summary', key)] = value
        for key, value in row['overall'].items():
            if not isinstance(value, list):
                flat[(method, 'overall', key)] = value
        for k, metrics in row['clients'].items():
            for key, value in metrics.items():
                if not isinstance(value, list):
                    flat[(method, k, key)] = value
    return flat


def interval(values):
    values = np.array([v for v in values if v is not None and np.isfinite(v)], float)
    if values.size < 50:
        return None
    return np.quantile(values, [0.025, 0.975]).tolist()


# ------------------------------------------------------------------ main analysis

def analyse(scheme, groups, frame, split, config, args, rng):
    started = time.time()
    names = ('train', 'validation', 'test')
    masks = {s: split == s for s in names}
    data = {s: frame.loc[masks[s]] for s in names}
    y = {s: data[s][LABEL].to_numpy(dtype=int) for s in names}
    c = {s: groups[masks[s]] for s in names}
    x = {s: np.asarray(features(data[s]), dtype=float) for s in names}
    clients = sorted(np.unique(c['train']))

    counts = {s: {k: dict(patients=int((c[s] == k).sum()), events=int(y[s][c[s] == k].sum()))
                  for k in clients} for s in names}

    lr = float(config.get('learning_rate', config.get('lr', config.get('eta', 0.5))))
    rounds = int(config.get('rounds', 100))
    steps = int(config['local_steps'])

    train_clients = [(x['train'][c['train'] == k], y['train'][c['train'] == k]) for k in clients]
    w_fed, _ = fit(train_clients, (x['validation'], y['validation']), config, steps, None, None, 0,
                   np.random.default_rng(SEED))
    w_pool, _ = fit([(x['train'], y['train'])], (x['validation'], y['validation']), config, steps, None, None, 0,
                    np.random.default_rng(SEED))
    w_own, _ = cc.fedavg_personal(train_clients, lr, rounds, steps, personal=False)
    w_pfl, b_pfl = cc.fedavg_personal(train_clients, lr, rounds, steps, personal=True)
    b_map = dict(zip(clients, b_pfl))
    w_local = {k: cc.local_only(xk, yk, lr, rounds * steps) for k, (xk, yk) in zip(clients, train_clients)}

    def logits(s):
        eta = x[s] @ np.asarray(w_fed)
        eta_pfl = x[s] @ w_pfl + np.vectorize(b_map.get)(c[s])
        eta_pool = x[s] @ np.asarray(w_pool)
        eta_local = np.array([x[s][i] @ w_local[k] for i, k in enumerate(c[s])])
        return eta, eta_pfl, eta_pool, eta_local

    L = {s: logits(s) for s in names}

    # Proposition 1 check on training data: client mean residual p - y.
    residuals = {}
    for k in clients:
        m = c['train'] == k
        residuals[k] = dict(
            fedavg=float(np.mean(cc.sigmoid(L['train'][0][m]) - y['train'][m])),
            personalized=float(np.mean(cc.sigmoid(L['train'][1][m]) - y['train'][m])))
    implementation_check = float(np.max(np.abs(x['test'] @ w_own - L['test'][0])))

    def run_once(val_idx, test_idx):
        ev, yv, cv = L['validation'][0][val_idx], y['validation'][val_idx], c['validation'][val_idx]
        params, fb = update_maps(ev, yv, cv, L['validation'][1][val_idx], clients)
        eta, eta_pfl, eta_pool, eta_local = (arr[test_idx] for arr in L['test'])
        preds = {m: predict(m, params, eta, eta_pfl, eta_pool, eta_local, c['test'][test_idx]) for m in UPDATES}
        return params, fb, preds

    full_val = np.arange(y['validation'].size)
    full_test = np.arange(y['test'].size)
    params, fallbacks, preds = run_once(full_val, full_test)
    point = evaluate_all(preds, y['test'], c['test'], clients)

    # Stratified bootstrap: validation and test patients resampled within client,
    # updating maps refitted in every draw; training-era fits held fixed.
    val_by = {k: np.flatnonzero(c['validation'] == k) for k in clients}
    test_by = {k: np.flatnonzero(c['test'] == k) for k in clients}
    draws, recal_draws, fallback_counts = [], [], {}
    for _ in range(args.bootstrap):
        vi = np.concatenate([rng.choice(v, v.size) for v in val_by.values()])
        ti = np.concatenate([rng.choice(t, t.size) for t in test_by.values()])
        p_b, fb, preds_b = run_once(vi, ti)
        for key, used in fb.items():
            fallback_counts[key] = fallback_counts.get(key, 0) + int(used)
        recal_draws.append(p_b['Local logistic recalibration'])
        draws.append(flatten(evaluate_all(preds_b, y['test'][ti], c['test'][ti], clients, light=True)))

    intervals, paired = {}, {}
    point_flat = flatten(point)
    for key in point_flat:
        intervals['|'.join(key)] = interval([d.get(key) for d in draws])
    for method in UPDATES:
        if method == REFERENCE:
            continue
        for key in point_flat:
            if key[0] != method:
                continue
            ref_key = (REFERENCE,) + key[1:]
            diffs = [None if d.get(key) is None or d.get(ref_key) is None else d[key] - d[ref_key] for d in draws]
            est = None if point_flat[key] is None or point_flat.get(ref_key) is None else point_flat[key] - point_flat[ref_key]
            paired['|'.join(key)] = dict(estimate=est, interval=interval(diffs))
    # Secondary paired contrasts: local vs federated CITL (H2), local LR vs local CITL (H3),
    # personalized + local CITL vs local CITL (H5).
    for a_m, b_m in (('Local CITL', 'Federated CITL'),
                     ('Local logistic recalibration', 'Local CITL'),
                     ('Personalized FL + local CITL', 'Local CITL')):
        for key in point_flat:
            if key[0] != a_m:
                continue
            other = (b_m,) + key[1:]
            diffs = [None if d.get(key) is None or d.get(other) is None else d[key] - d[other] for d in draws]
            est = None if point_flat[key] is None or point_flat.get(other) is None else point_flat[key] - point_flat[other]
            paired[f'{a_m} minus {b_m}|' + '|'.join(key[1:])] = dict(estimate=est, interval=interval(diffs))

    slopes = {k: dict(intercept=params['Local logistic recalibration'][k][0],
                      slope=params['Local logistic recalibration'][k][1],
                      slope_interval=interval([d[k][1] for d in recal_draws]))
              for k in clients}

    # Learning curve for local updating on the client's own recent data.
    curve = {}
    for k in clients:
        idx = val_by[k]
        tm = c['test'] == k
        eta_t, y_t = L['test'][0][tm], y['test'][tm]
        base = cc.metrics(y_t, cc.sigmoid(eta_t), PRIMARY)
        curve[k] = {}
        for size in CURVE_SIZES:
            if size > idx.size:
                continue
            store = {name: [] for name in ('citl_abs_citl', 'citl_ici', 'lr_abs_citl', 'lr_ici',
                                           'citl_nb_gain_0.05', 'citl_nb_gain_0.1',
                                           'lr_nb_gain_0.05', 'lr_nb_gain_0.1')}
            undefined = lr_fallback = 0
            for _ in range(CURVE_REPS):
                s = rng.choice(idx, size, replace=False)
                ev, yv = L['validation'][0][s], y['validation'][s]
                a = cc.fit_offset(ev, yv)
                undefined += a is None
                a = 0.0 if a is None else a
                ab = cc.fit_logistic_recal(ev, yv) if yv.sum() >= LR_MIN_EVENTS else None
                lr_fallback += ab is None
                ab = (a, 1.0) if ab is None else ab
                for tag, z in (('citl', eta_t + a), ('lr', ab[0] + ab[1] * eta_t)):
                    m = cc.metrics(y_t, cc.sigmoid(z), PRIMARY)
                    store[f'{tag}_abs_citl'].append(None if m['citl'] is None else abs(m['citl']))
                    store[f'{tag}_ici'].append(m['ici'])
                    for t in PRIMARY:
                        store[f'{tag}_nb_gain_{t:g}'].append(m[f'nb_{t:g}'] - base[f'nb_{t:g}'])
            curve[k][str(size)] = dict(
                undefined_citl=int(undefined), lr_fallback=int(lr_fallback),
                **{name: dict(zip(('p2.5', 'median', 'p97.5'),
                                  np.quantile([v for v in vals if v is not None], [0.025, 0.5, 0.975]).tolist()))
                   for name, vals in store.items()})
        curve[k]['reference_global'] = dict(abs_citl=None if base['citl'] is None else abs(base['citl']),
                                            ici=base['ici'])

    return dict(scheme=scheme, clients=clients, counts=counts,
                optimisation=dict(learning_rate=lr, rounds=rounds, local_steps=steps,
                                  own_fedavg_vs_audited_max_abs_logit_diff=implementation_check),
                training_residuals=residuals,
                update_parameters=dict(
                    federated_citl=params['Federated CITL'],
                    federated_logistic=params['Federated logistic recalibration'],
                    local_logistic=slopes,
                    personalized_intercepts=dict(map(lambda kv: (kv[0], float(kv[1])), b_map.items())),
                    personalized_local_citl=params['Personalized FL + local CITL']),
                fallbacks_full_data={k: bool(v) for k, v in fallbacks.items()},
                fallback_counts_bootstrap=fallback_counts,
                point=point, intervals=intervals, paired=paired, learning_curve=curve,
                seconds=round(time.time() - started, 1))


def plot(report, out):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as error:  # plotting is optional
        print(f'Plot skipped: {error}')
        return
    for scheme, res in report['analyses'].items():
        clients = res['clients']
        methods = ('FedAvg global', 'Federated CITL', 'Local CITL', 'Local logistic recalibration')
        fig, axes = plt.subplots(1, len(clients), figsize=(4 * len(clients), 4), layout='constrained', squeeze=False)
        for ax, k in zip(axes[0], clients):
            for m in methods:
                b = res['point'][m]['clients'][k]['bins']
                ax.plot([i['mean_risk'] for i in b], [i['event_rate'] for i in b], 'o-', label=m)
            top = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([0, top], [0, top], '--', color='gray')
            ax.set(title=k, xlabel='Mean predicted risk', ylabel='Observed event fraction')
        axes[0, 0].legend(fontsize=7)
        fig.savefig(out / f'reliability_{scheme}.png', dpi=240)
        fig.savefig(out / f'reliability_{scheme}.pdf')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 0.5 + 0.45 * len(clients) * 3), layout='constrained')
        ypos = 0
        labels = []
        for k in clients:
            for m in ('FedAvg global', 'Federated CITL', 'Local CITL'):
                est = res['point'][m]['clients'][k]['citl']
                ci = res['intervals'].get(f'{m}|{k}|citl')
                if est is not None:
                    ax.plot(est, ypos, 'o', color={'FedAvg global': 'C0', 'Federated CITL': 'C1', 'Local CITL': 'C2'}[m])
                    if ci:
                        ax.plot(ci, [ypos, ypos], '-', color='gray')
                labels.append(f'{k} · {m}')
                ypos += 1
        ax.axvline(0, color='black', lw=0.8)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.invert_yaxis()
        ax.set(xlabel='Calibration-in-the-large on the test era (logit scale)')
        fig.savefig(out / f'citl_forest_{scheme}.png', dpi=240)
        fig.savefig(out / f'citl_forest_{scheme}.pdf')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--cohort', type=Path, default=Path('local_output/audited_cohort/cohort.csv'))
    parser.add_argument('--config', type=Path, default=Path('local_output/experiments.json'))
    parser.add_argument('--cohort-sha256', default=COHORT_HASH)
    args = parser.parse_args()
    if args.out.exists() or args.bootstrap < 200:
        parser.error('Use a fresh output directory and at least 200 bootstrap draws')
    if sha256(args.cohort.read_bytes()).hexdigest() != args.cohort_sha256:
        raise ValueError('Audited cohort fingerprint mismatch')
    frame = pd.read_csv(args.cohort)
    index, _ = extract_index(args.root)
    index = index.reset_index(drop=True)
    fields = ['age_years', 'gender', 'split', LABEL, 'admission_type',
              'admission_location', 'insurance', 'race', 'hf_principal']
    pd.testing.assert_frame_equal(frame[fields], index[fields], check_dtype=False)
    if not index.subject_id.is_unique:
        raise ValueError('Duplicate index patient')
    split = temporal_split(args.root, index, 2017, 2020).to_numpy()
    labels = frame[LABEL].to_numpy(dtype=int)
    config = json.loads(args.config.read_text(encoding='utf-8'))

    raw_service = landmark_service(args.root, index)
    service = np.array([SERVICE_GROUP.get(s, OTHER) for s in raw_service], dtype=object)
    service, service_log = merge_small(service, labels, split, service_fallback)
    age_icu = np.array([f"{'Age 65+' if a >= 65 else 'Age <65'} / {'early ICU' if i > 0 else 'no early ICU'}"
                        for a, i in zip(frame.age_years, frame.icu_stays_24h)], dtype=object)
    age_icu, age_log = merge_small(age_icu, labels, split, largest_fallback)

    used = split != 'excluded'
    codes, code_counts = np.unique(raw_service[used], return_counts=True)
    code_table = {str(k): int(v) for k, v in zip(codes, code_counts) if v >= MIN_CELL}
    code_table['codes with fewer than 30 patients (pooled)'] = int(code_counts[code_counts < MIN_CELL].sum())

    report = dict(cohort_sha256=args.cohort_sha256, seed=SEED, bootstrap_draws=args.bootstrap,
                  thresholds=THRESHOLDS, primary_thresholds=PRIMARY, min_cell=MIN_CELL,
                  years=[2017, 2020], service_codes_at_landmark=code_table,
                  merges=dict(service=service_log, age_icu=age_log), analyses={})
    rng = np.random.default_rng(SEED)
    for scheme, groups in (('service', service), ('age_icu', age_icu)):
        print(f'Running {scheme} clients', flush=True)
        report['analyses'][scheme] = analyse(scheme, groups, frame, split, config, args, rng)
    args.out.mkdir(parents=True)
    (args.out / 'client_calibration_results.json').write_text(
        json.dumps(report, indent=2, allow_nan=False, default=float), encoding='utf-8')
    plot(report, args.out)
    print(f'Wrote aggregate results to {args.out}')


if __name__ == '__main__':
    main()
