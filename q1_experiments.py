"""Repeated fixed-holdout heart failure experiments."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from shutil import copy2

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
import pandas as pd

from run_experiments import LABEL, features, metrics, sigmoid

ALPHAS = (0.3, 0.5, 1.0, 2.0)


def partition(labels, rng, sites, alpha):
    iid = np.array_split(rng.permutation(len(labels)), sites)
    for attempt in range(1, 1001):
        groups = [[] for _ in iid]
        for label in (0, 1):
            rows = rng.permutation(np.flatnonzero(labels == label))
            counts = rng.multinomial(len(rows), rng.dirichlet(np.full(sites, alpha)))
            for group, chunk in zip(groups, np.split(rows, counts.cumsum()[:-1])):
                group.extend(chunk)
        if all(groups):
            noniid = [np.asarray(group) for group in groups]
            for partition_groups in (iid, noniid):
                if not np.array_equal(np.sort(np.concatenate(partition_groups)), np.arange(len(labels))):
                    raise ValueError('Partition must cover each training patient once')
            return iid, noniid, attempt
    raise ValueError('Unable to generate nonempty clients')


def fit(clients, validation, config, steps, clip, noise, prox, rng):
    total_n = sum(len(labels) for _, labels in clients)
    weights = np.zeros(validation[0].shape[1])
    norms = [np.linalg.norm(matrix, axis=1) for matrix, _ in clients] if clip else None
    history = []
    for _ in range(config['rounds']):
        total = np.zeros_like(weights)
        for index, (matrix, labels) in enumerate(clients):
            if steps == 1:
                residual = sigmoid(matrix @ weights) - labels
                if clip:
                    residual *= clip / np.maximum(clip, np.abs(residual) * norms[index])
                total += matrix.T @ residual
            else:
                local = weights.copy()
                for _ in range(steps):
                    gradient = matrix.T @ (sigmoid(matrix @ local) - labels) / len(labels)
                    local -= config['learning_rate'] * (gradient + prox * (local - weights))
                total += len(labels) * (weights - local)
        if noise:
            total += rng.normal(0, 2 * clip * noise, len(weights))
        scale = config['learning_rate'] if steps == 1 else 1
        weights -= scale * total / total_n
        logits = validation[0] @ weights
        history.append(float(np.mean(np.logaddexp(0, logits) - validation[1] * logits)))
    return weights, np.array(history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('local_output'))
    parser.add_argument('--out', type=Path, default=Path('local_output/q1_experiments'))
    parser.add_argument('--alpha-study', action='store_true')
    args = parser.parse_args()
    config = json.loads((args.input / 'experiments.json').read_text())
    if args.alpha_study:
        config['alpha_study'] = ALPHAS
    cohort = args.input / 'cohort.csv'
    frame = pd.read_csv(cohort)
    manifest = json.loads((args.input / 'manifest.json').read_text())
    if len(frame) != manifest['counts']['index_patients'] or not frame.split.isin(config['split_sizes']).all():
        raise ValueError('Export does not match cohort metadata')
    matrix = features(frame)
    subsets = {}
    for name, expected in config['split_sizes'].items():
        mask = frame.split.eq(name)
        subsets[name] = matrix[mask], frame.loc[mask, LABEL].to_numpy()
        if mask.sum() != expected or set(subsets[name][1]) != {0, 1}:
            raise ValueError(f'Unexpected {name} cohort')
    train_x, train_y = subsets['train']
    test_x, test_y = subsets['test']
    seeds = list(range(config['seed'], config['seed'] + 10))
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'runs.csv').exists():
        raise FileExistsError('Use a fresh output directory')
    for source in [Path(__file__), Path(__file__).with_name('run_experiments.py'),
                   Path(__file__).with_name('build_cohort.py'),
                   args.input / 'experiments.json', Path(__file__).with_name('q1_report.py')]:
        destination = args.out / source.name
        if source.resolve() != destination.resolve():
            copy2(source, destination)
    runs, traces, allocations, predictions = [], [], [], {}
    steps = config['local_steps']
    clip = config['dp']['clip']
    baseline_noise = config['dp']['noise_multiplier']
    for seed in seeds:
        iid, noniid, attempts = partition(train_y, np.random.default_rng(seed), config['sites'], config['alpha'])
        partitions = {config['alpha']: (noniid, attempts)}
        for alpha in config.get('alpha_study', ()):
            if alpha not in partitions:
                _, groups, retries = partition(train_y, np.random.default_rng(seed), config['sites'], alpha)
                partitions[alpha] = groups, retries
        for alpha, (groups, draws) in [(None, (iid, 1)), *partitions.items()]:
            for client, group in enumerate(groups, 1):
                events = int(train_y[group].sum())
                allocations.append(dict(seed=seed, scheme='IID' if alpha is None else 'Non-IID',
                                        alpha=alpha, client=client, patients=len(group), events=events,
                                        mortality_rate=events / len(group), attempts=draws,
                                        small_client=len(group) < 100, single_class=events in (0, len(group))))
        models = [
            ('Centralized', [np.arange(len(train_y))], steps, None, None, 0),
            ('FedAvg IID', iid, steps, None, None, 0),
            ('FedAvg non-IID', noniid, steps, None, None, 0),
            ('FedProx non-IID', noniid, steps, None, None, 0.1),
            ('FedSGD non-IID', noniid, 1, None, None, 0),
            ('Clipped FedSGD', noniid, 1, clip, None, 0),
            *[(f'Noisy FedSGD (m={multiplier:g})', noniid, 1, clip, multiplier, 0)
              for multiplier in (baseline_noise / 2, baseline_noise, baseline_noise * 2)],
        ]
        if args.alpha_study:
            models.extend(
                (f'{method} alpha={alpha:g}', groups, steps, None, None, prox)
                for alpha, (groups, _) in partitions.items() if alpha != config['alpha']
                for method, prox in [('FedAvg', 0), ('FedProx', 0.1)]
            )
        for model, groups, local_steps, bound, noise, prox in models:
            clients = [(train_x[group], train_y[group]) for group in groups]
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(1000 * (noise or 0)), 817]))
            weights, loss = fit(clients, subsets['validation'], config, local_steps, bound, noise, prox, rng)
            probability = sigmoid(test_x @ weights)
            predictions.setdefault(model, []).append(probability)
            within = np.flatnonzero(loss <= loss.min() + 0.001)
            convergence = int(within[0] + 1)
            per_direction = 0 if len(groups) == 1 else len(groups) * weights.nbytes
            rounds = config['rounds']
            runs.append(dict(seed=seed, model=model, **metrics(test_y, probability),
                             rounds=rounds, local_steps=local_steps, clip=bound, noise_multiplier=noise,
                             prox=prox, convergence_checkpoint=convergence,
                             upload_bytes=per_direction * rounds, download_bytes=per_direction * rounds,
                             total_bytes=2 * per_direction * rounds,
                             bytes_at_checkpoint=2 * per_direction * convergence))
            traces.extend(dict(seed=seed, model=model, checkpoint=index, validation_log_loss=value)
                          for index, value in enumerate(loss, 1))
        print(f'Finished seed {seed}', flush=True)
    for name, data in [('runs', runs), ('convergence', traces), ('clients', allocations)]:
        pd.DataFrame(data).to_csv(args.out / f'{name}.csv', index=False)
    metadata = dict(seeds=seeds, cohort_sha256=hashlib.sha256(cohort.read_bytes()).hexdigest(),
                    splits={name: dict(patients=len(labels), events=int(labels.sum()))
                            for name, (_, labels) in subsets.items()},
                    parameters=matrix.shape[1], numpy=np.__version__, pandas=pd.__version__)
    from q1_report import report
    report(args.out, config, metadata, test_y, predictions)


if __name__ == '__main__':
    main()
