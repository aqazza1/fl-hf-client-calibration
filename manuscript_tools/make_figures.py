"""Manuscript figures from the aggregate results JSON (no patient-level data)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

src = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(exist_ok=True)
r = json.loads(src.read_text(encoding='utf-8'))

plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'legend.frameon': False})
STYLE = {  # reference in neutral gray, updates in fixed categorical order; marker = secondary encoding
    'FedAvg global': ('#6b6b66', 'o', 'Global FedAvg'),
    'Federated CITL': ('#2a78d6', 's', 'Federated CITL'),
    'Local CITL': ('#eb6834', 'D', 'Local CITL'),
    'Local logistic recalibration': ('#1baf7a', '^', 'Local logistic recalibration'),
}
TITLES = {'service': 'Hospital-service clients', 'age_icu': 'Age-by-early-ICU clients'}


def label(scheme, k):
    if scheme == 'age_icu' and k == 'Age 65+ / no early ICU':
        return 'No early ICU\n(all ages, merged)'
    return k.replace(' / ', '\n')


# Figure 1: client calibration-in-the-large with 95% intervals.
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), layout='constrained', sharex=True)
for ax, (scheme, a) in zip(axes, r['analyses'].items()):
    clients = a['clients']
    methods = list(STYLE)
    for i, k in enumerate(clients):
        for j, m in enumerate(methods):
            y = i + (j - 1.5) * 0.17
            est = a['point'][m]['clients'][k]['citl']
            lo, hi = a['intervals'][f'{m}|{k}|citl']
            color, marker, name = STYLE[m]
            ax.plot([lo, hi], [y, y], color=color, lw=1.2, solid_capstyle='round')
            ax.plot(est, y, marker=marker, color=color, ms=5, mec='white', mew=0.8,
                    label=name if i == 0 else None, ls='none')
    ax.axvline(0, color='#3a3a38', lw=0.6)
    ax.set_yticks(range(len(clients)), [label(scheme, k) for k in clients])
    ax.invert_yaxis()
    ax.set_title(TITLES[scheme], fontsize=9.5, loc='left')
    ax.grid(axis='x', color='#e4e3df', lw=0.5)
    ax.set_axisbelow(True)
fig.supxlabel('Calibration-in-the-large in the test era (logit scale; 0 = calibrated, > 0 = risk underestimated)', fontsize=9)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside upper center', ncol=4, fontsize=8)
fig.savefig(out / 'Fig1.png', dpi=300)
fig.savefig(out / 'Fig1.pdf')
plt.close(fig)

# Figure 2: reliability by service client.
a = r['analyses']['service']
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), layout='constrained', sharey=True)
LIM = 0
for k in a['clients']:
    for m in ('FedAvg global', 'Federated CITL', 'Local logistic recalibration'):
        for v in a['point'][m]['clients'][k]['bins']:
            LIM = max(LIM, v['mean_risk'], v['event_rate'])
LIM = min(1, LIM * 1.08)
for ax, k in zip(axes, a['clients']):
    top = 0
    for m in ('FedAvg global', 'Federated CITL', 'Local logistic recalibration'):
        b = a['point'][m]['clients'][k]['bins']
        x = [v['mean_risk'] for v in b]
        y = [v['event_rate'] for v in b]
        top = max(top, max(x + y))
        color, marker, name = STYLE[m]
        ax.plot(x, y, color=color, lw=1.5, marker=marker, ms=5, mec='white', mew=0.8, label=name)
    ax.plot([0, LIM], [0, LIM], ls='--', color='#9a9994', lw=0.8)
    ax.set_xlim(0, LIM)
    ax.set_ylim(0, LIM)
    ax.set_title(k, fontsize=9.5, loc='left')
    ax.set_xlabel('Mean predicted risk')
    ax.grid(color='#e4e3df', lw=0.5)
    ax.set_axisbelow(True)
axes[0].set_ylabel('Observed death rate')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=8)
fig.savefig(out / 'Fig2.png', dpi=300)
fig.savefig(out / 'Fig2.pdf')
plt.close(fig)

# Figure 3: local sample size versus worst calibration error after local CITL.
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), layout='constrained', sharex=True)
for row, (scheme, a) in zip(axes, r['analyses'].items()):
    for ax, k in zip(row, a['clients']):
        c = a['learning_curve'][k]
        sizes = sorted(int(s) for s in c if s != 'reference_global')
        med = [c[str(s)]['citl_abs_citl']['median'] for s in sizes]
        lo = [c[str(s)]['citl_abs_citl']['p2.5'] for s in sizes]
        hi = [c[str(s)]['citl_abs_citl']['p97.5'] for s in sizes]
        ax.fill_between(sizes, lo, hi, color='#eb6834', alpha=0.18, lw=0)
        ax.plot(sizes, med, color='#eb6834', marker='D', ms=4, lw=1.5, label='Local CITL (median, 95% range)')
        ref = c['reference_global']['abs_citl']
        ax.axhline(ref, color='#6b6b66', ls='--', lw=1, label='Global FedAvg, no update')
        ax.set_xscale('log')
        ax.minorticks_off()
        ax.set_ylim(0, 1.9)
        ax.set_title(label(scheme, k).replace('\n', ' '), fontsize=9, loc='left')
        ax.grid(color='#e4e3df', lw=0.5)
        ax.set_axisbelow(True)
    row[0].set_ylabel('|Calibration-in-the-large|')
for ax in axes.flat:
    ax.set_xticks([25, 50, 100, 200, 400, 800], ['25', '50', '100', '200', '400', '800'])
    ax.minorticks_off()
for ax in axes[1]:
    ax.set_xlabel('Updating-era patients used')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside lower center', ncol=2, fontsize=8)
fig.savefig(out / 'Fig3.png', dpi=300)
fig.savefig(out / 'Fig3.pdf')
plt.close(fig)
print('figures written')
