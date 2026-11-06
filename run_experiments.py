"""Local logistic-regression baselines on identical patient holdouts."""

import argparse
from hashlib import sha256
import json
from math import log, sqrt
from pathlib import Path
from random import SystemRandom

import numpy as np
import pandas as pd

from build_cohort import LABEL, LABS, SEED, VITALS


def features(frame):
    # Fixed transforms avoid learning any preprocessing statistics from private data.
    columns = [np.ones(len(frame)), frame.age_years.clip(18, 89).to_numpy() / 89]
    scales = {'icu_hours_24h': 24, 'icu_stays_24h': 4}
    for name, _, _, bound in (list(LABS.values()) + list(VITALS.values())):
        scales[f'{name}_count'] = 100
        for stat in ('mean', 'min', 'max'):
            scales[f'{name}_{stat}'] = 50 if name == 'temperature_c' else bound
    for name, scale in scales.items():
        if name in frame:
            value = frame[name]
            columns.extend([value.fillna(0).clip(0, scale).to_numpy() / scale, value.isna().to_numpy()])
    for name in ('gender', 'admission_type', 'admission_location'):
        values = frame[name].fillna('').astype(str)
        buckets = values.map(lambda value: sha256(value.encode()).digest()[0] % 16).to_numpy()
        columns.extend(buckets == bucket for bucket in range(16))
    return np.column_stack(columns).astype(np.float64)


def partition(y, sites, alpha):
    rng = np.random.default_rng(SEED)
    iid = np.array_split(rng.permutation(len(y)), sites)
    noniid = [[] for _ in range(sites)]
    for label in (0, 1):
        rows = rng.permutation(np.flatnonzero(y == label))
        allocation = rng.multinomial(len(rows), rng.dirichlet(np.full(sites, alpha)))
        for client, group in zip(noniid, np.split(rows, allocation.cumsum()[:-1])):
            client.extend(group)
    noniid = [np.asarray(rows, dtype=int) for rows in noniid]
    if any(len(rows) == 0 for rows in noniid):
        raise ValueError('Empty simulated client. Use fewer clients or a larger alpha.')
    return iid, noniid


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def train(x, y, groups, rounds, steps, lr, clip=None, noise=None):
    weights = np.zeros(x.shape[1])
    clients = [(x[group], y[group]) for group in groups]
    random = SystemRandom()
    for _ in range(rounds):
        if clip is None:
            total = np.zeros_like(weights)
            for local_x, local_y in clients:
                local = weights.copy()
                for _ in range(steps):
                    residual = sigmoid(local_x @ local) - local_y
                    local -= lr * (local_x.T @ residual) / len(local_y)
                total += len(local_y) * local
            weights = total / len(y)
        else:
            total = np.zeros_like(weights)
            for local_x, local_y in clients:
                residual = sigmoid(local_x @ weights) - local_y
                norm = np.abs(residual) * np.linalg.norm(local_x, axis=1)
                factor = clip / np.maximum(clip, norm)
                total += local_x.T @ (residual * factor)
            # Replace-one sensitivity of the sum of clipped patient gradients is 2C.
            if noise is not None:
                total += np.array([random.gauss(0, 2 * clip * noise) for _ in weights])
            weights -= lr * total / len(y)
    return weights


def metrics(y, probability):
    order = np.argsort(-probability)
    sorted_p = probability[order]
    ends = np.r_[np.flatnonzero(sorted_p[:-1] != sorted_p[1:]), len(y) - 1]
    tp = np.cumsum(y[order])[ends]
    fp = ends + 1 - tp
    recall = np.r_[0, tp / tp[-1]]
    fpr = np.r_[0, fp / fp[-1]]
    return {
        'auroc': float(np.sum(np.diff(fpr) * (recall[:-1] + recall[1:]) / 2)),
        'average_precision': float(np.sum(np.diff(recall) * tp / (ends + 1))),
        'brier': float(np.mean((y - probability) ** 2)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New local JSON file')
    parser.add_argument('--sites', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--local-steps', type=int, default=5)
    parser.add_argument('--noise-multiplier', type=float, default=10)
    args = parser.parse_args()
    if (args.sites < 2 or min(args.rounds, args.local_steps) < 1 or
            not np.isfinite(args.alpha) or not np.isfinite(args.noise_multiplier) or
            min(args.alpha, args.noise_multiplier) <= 0 or args.out.exists()):
        parser.error('Use positive settings, at least two clients, and a new output file')
    frame = pd.read_csv(args.cohort)
    all_x = features(frame)
    all_y = frame[LABEL].to_numpy()
    subsets = {split: frame.split.eq(split).to_numpy() for split in ('train', 'validation', 'test')}
    if any(set(all_y[mask]) != {0, 1} for mask in subsets.values()):
        parser.error('Every split needs both outcome classes')
    x, y = all_x[subsets['train']], all_y[subsets['train']]
    if len(y) < args.sites:
        parser.error('Training patients must outnumber clients')
    iid, noniid = partition(y, args.sites, args.alpha)
    rho = args.rounds / (2 * args.noise_multiplier ** 2)
    delta = min(1e-6, 1 / len(y) ** 2)
    lr, clip = 0.5, 1.0
    report = {
        'seed': SEED, 'sites': args.sites, 'alpha': args.alpha, 'rounds': args.rounds,
        'local_steps': args.local_steps, 'learning_rate': lr,
        'privacy_scope': 'Ideal single-fit reference for a fixed-size clipped sum with a trusted '
                         'aggregator. Exact routing summaries and metrics are outside that reference.',
        'dp': {'ideal_single_fit_epsilon': rho + 2 * sqrt(rho * log(1 / delta)),
               'formal_dp': False, 'secure_aggregation': False, 'delta': delta,
               'clip': clip, 'noise_multiplier': args.noise_multiplier},
        'split_sizes': {name: int(mask.sum()) for name, mask in subsets.items()},
        'partitions': {name: [{'n': len(group), 'events': int(y[group].sum())} for group in groups]
                       for name, groups in [('iid', iid), ('non_iid', noniid)]},
        'metrics': {},
    }
    models = [
        ('centralized', [np.arange(len(y))], None, None, args.local_steps),
        ('fedavg_iid', iid, None, None, args.local_steps),
        ('fedavg_non_iid', noniid, None, None, args.local_steps),
        ('clipped_fedsgd_non_iid', noniid, clip, None, 1),
        ('noisy_fedsgd_non_iid', noniid, clip, args.noise_multiplier, 1),
    ]
    for name, groups, bound, noise, steps in models:
        weights = train(x, y, groups, args.rounds, steps, lr, bound, noise)
        report['metrics'][name] = {
            split: metrics(all_y[mask], sigmoid(all_x[mask] @ weights))
            for split, mask in subsets.items() if split != 'train'
        }
        print(f'Completed {name}', flush=True)
    with args.out.open('x', encoding='utf-8') as output:
        json.dump(report, output, indent=2)


if __name__ == '__main__':
    main()
