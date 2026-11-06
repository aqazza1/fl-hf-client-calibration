"""Markdown tables for the manuscript, generated from the aggregate results JSON."""
import json
import sys
from pathlib import Path

r = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
NAMES = {'FedAvg global': 'Global FedAvg (reference)', 'Federated CITL': 'Federated CITL',
         'Federated logistic recalibration': 'Federated logistic recalibration',
         'Local CITL': 'Local CITL', 'Local logistic recalibration': 'Local logistic recalibration',
         'Personalized FL': 'Personalized FL', 'Personalized FL + local CITL': 'Personalized FL + local CITL',
         'Pooled': 'Pooled (central)', 'Local-only': 'Local-only'}


def client(scheme, k):
    if scheme == 'age_icu' and k == 'Age 65+ / no early ICU':
        return 'No early ICU, all ages (merged)'
    if scheme == 'service' and k == 'Other services':
        return 'Other services (incl. CV and thoracic surgery)'
    return k.replace(' / ', ', ')


def n(x, d=3):
    return '–' if x is None else f'{x:.{d}f}'


def ci(est, iv, d=3, sign=False):
    if est is None:
        return '–'
    f = f'{{:+.{d}f}}' if sign else f'{{:.{d}f}}'
    if not iv:
        return f.format(est)
    return f'{f.format(est)} ({f.format(iv[0])} to {f.format(iv[1])})'


out = {}
A = r['analyses']

# Table 1: counts by era.
rows = ['| Partition | Client | Development (patients/deaths) | Updating (patients/deaths) | Test (patients/deaths) |',
        '|---|---|---|---|---|']
for scheme, label in (('service', 'Hospital service'), ('age_icu', 'Age × early ICU')):
    a = A[scheme]
    for k in a['clients']:
        cells = [f"{a['counts'][e][k]['patients']:,} / {a['counts'][e][k]['events']}" for e in ('train', 'validation', 'test')]
        rows.append(f'| {label} | {client(scheme, k)} | ' + ' | '.join(cells) + ' |')
out['T1'] = '\n'.join(rows)

# Table 2: global model by client in the test era.
rows = ['| Partition | Client | AUROC | AP (prevalence) | Brier | CITL (95% CI) | Slope | O/E | ICI |',
        '|---|---|---|---|---|---|---|---|---|']
for scheme, label in (('service', 'Hospital service'), ('age_icu', 'Age × early ICU')):
    a = A[scheme]
    m = 'FedAvg global'
    for k in a['clients']:
        c = a['point'][m]['clients'][k]
        rows.append(f"| {label} | {client(scheme, k)} | {n(c['auroc'])} | {n(c['average_precision'])} ({c['events']/c['n']:.3f}) | "
                    f"{n(c['brier'], 4)} | {ci(c['citl'], a['intervals'][f'{m}|{k}|citl'], 2)} | {n(c['slope'], 2)} | {n(c['o_e'], 2)} | {n(c['ici'], 4)} |")
o = A['service']['point']['FedAvg global']['overall']
rows.append(f"| All | All test patients | {n(o['auroc'])} | {n(o['average_precision'])} ({o['events']/o['n']:.3f}) | {n(o['brier'], 4)} | "
            f"{ci(o['citl'], A['service']['intervals']['FedAvg global|overall|citl'], 2)} | {n(o['slope'], 2)} | {n(o['o_e'], 2)} | – |")
out['T2'] = '\n'.join(rows)

# Tables 3 and 4: strategies by partition.
for key, scheme in (('T3', 'service'), ('T4', 'age_icu')):
    a = A[scheme]
    rows = ['| Strategy | Worst-client CITL (95% CI) | Difference from global (95% CI) | Weighted ICI | Difference in ICI (95% CI) | AUROC | NB at 10% | Difference in NB at 10% (95% CI) |',
            '|---|---|---|---|---|---|---|---|']
    for m, name in NAMES.items():
        s = a['point'][m]['summary']
        o = a['point'][m]['overall']
        if m == 'FedAvg global':
            d1 = d2 = d3 = '–'
        else:
            p = a['paired']
            d1 = ci(p[f'{m}|summary|max_abs_citl']['estimate'], p[f'{m}|summary|max_abs_citl']['interval'], 3, True)
            d2 = ci(p[f'{m}|summary|weighted_ici']['estimate'], p[f'{m}|summary|weighted_ici']['interval'], 4, True)
            d3 = ci(p[f'{m}|overall|nb_0.1']['estimate'], p[f'{m}|overall|nb_0.1']['interval'], 4, True)
        rows.append(f"| {name} | {ci(s['max_abs_citl'], a['intervals'][f'{m}|summary|max_abs_citl'], 3)} | {d1} | "
                    f"{n(s['weighted_ici'], 4)} | {d2} | {n(o['auroc'])} | {n(o['nb_0.1'], 4)} | {d3} |")
    out[key] = '\n'.join(rows)

# Table 5: decisions by client, both thresholds in one row per strategy.
SHORT = {'FedAvg global': 'Global', 'Federated CITL': 'Federated CITL', 'Local logistic recalibration': 'Local LR'}
rows = ['| Client | Strategy | 5%: flagged / sensitivity | 5%: NB flag-all | 5%: NB model | 5%: difference (95% CI) | 10%: flagged / sensitivity | 10%: NB flag-all | 10%: NB model | 10%: difference (95% CI) |',
        '|---|---|---|---|---|---|---|---|---|---|']
for scheme in ('service', 'age_icu'):
    a = A[scheme]
    for k in a['clients']:
        for m in ('FedAvg global', 'Federated CITL', 'Local logistic recalibration'):
            c = a['point'][m]['clients'][k]
            cells = []
            for t in ('0.05', '0.1'):
                if m == 'FedAvg global':
                    d = '–'
                else:
                    p = a['paired'][f'{m}|{k}|nb_{t}']
                    d = ci(p['estimate'], p['interval'], 4, True)
                cells += [f"{c[f'flag_rate_{t}']:.2f} / {n(c[f'sens_{t}'], 2)}", f"{c[f'nb_all_{t}']:.4f}", f"{c[f'nb_{t}']:.4f}", d]
            name = client(scheme, k) if m == 'FedAvg global' else ''
            rows.append(f'| {name} | {SHORT[m]} | ' + ' | '.join(cells) + ' |')
out['T5'] = '\n'.join(rows)

# Table 6: local logistic recalibration estimated on the updating era.
rows = ['| Partition | Client | Intercept a_k | Slope b_k (95% CI) | Personalized intercept b_k (development) |', '|---|---|---|---|---|']
for scheme, label in (('service', 'Hospital service'), ('age_icu', 'Age × early ICU')):
    a = A[scheme]
    u = a['update_parameters']
    for k in a['clients']:
        v = u['local_logistic'][k]
        rows.append(f"| {label} | {client(scheme, k)} | {v['intercept']:.2f} | {ci(v['slope'], v['slope_interval'], 2)} | {u['personalized_intercepts'][k]:+.2f} |")
out['T6'] = '\n'.join(rows)

out = {k: v.replace(' -', ' \u2212').replace('(-', '(\u2212').replace('/ -', '/ \u2212') for k, v in out.items()}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
for k, v in out.items():
    print(k, v.count('\n') + 1, 'lines')
print(out['T5'][:900])
