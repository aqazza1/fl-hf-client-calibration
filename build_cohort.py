"""Build a local, 24-hour landmark cohort from MIMIC-IV CSV files."""

import argparse
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd


# itemid: (feature, accepted source units, lower bound, upper bound)
LABS = {
    50912: ('creatinine_mg_dl', ['mg/dl'], 0, 150),
    51006: ('bun_mg_dl', ['mg/dl'], 0, 300),
    50983: ('sodium_meq_l', ['meq/l', 'mmol/l'], 0, 200),
    50971: ('potassium_meq_l', ['meq/l', 'mmol/l'], 0, 30),
    50882: ('bicarbonate_meq_l', ['meq/l', 'mmol/l'], 0, 100),
    50931: ('glucose_mg_dl', ['mg/dl'], 0, 10000),
    50862: ('albumin_g_dl', ['g/dl'], 0, 10),
    51222: ('hemoglobin_g_dl', ['g/dl'], 0, 30),
}
VITALS = {
    220045: ('heart_rate_bpm', [], 0, 300),
    220179: ('noninvasive_sbp_mmhg', [], 0, 400),
    220181: ('noninvasive_map_mmhg', [], 0, 300),
    220210: ('respiratory_rate_min', [], 0, 70),
    220277: ('spo2_pct', [], 0, 100),
    223762: ('temperature_c', [], 10, 50),
    223761: ('temperature_c', [], 70, 120),
}
KEY = ['subject_id', 'hadm_id']
LABEL = 'hospital_death_after_24h'
SEED = 20261105


def read(root, table, columns, **kwargs):
    return pd.read_csv(root / f'{table}.csv.gz', usecols=columns, **kwargs)


def assign_split(subject_id):
    digest = hashlib.sha256(f'{SEED}:{subject_id}'.encode()).digest()
    bucket = int.from_bytes(digest[:8], 'big') % 100
    return 'train' if bucket < 70 else 'validation' if bucket < 85 else 'test'


def extract_index(root):
    dx = read(root, 'hosp/diagnoses_icd', KEY + ['seq_num', 'icd_version', 'icd_code'],
              dtype={'icd_code': 'string'})
    code = dx.icd_code.str.strip().str.upper().str.replace('.', '', regex=False)
    hf = ((dx.icd_version.eq(9) & code.str.startswith('428')) |
          (dx.icd_version.eq(10) & code.str.startswith('I50'))).fillna(False)
    dx = dx.loc[hf, KEY + ['seq_num']].assign(hf_principal=lambda x: x.seq_num.eq(1))
    dx = dx.groupby(KEY, as_index=False).hf_principal.max()
    admissions = read(root, 'hosp/admissions', KEY + [
        'admittime', 'dischtime', 'deathtime', 'hospital_expire_flag', 'admission_type',
        'admission_location', 'race', 'insurance'],
        parse_dates=['admittime', 'dischtime', 'deathtime'])
    patients = read(root, 'hosp/patients', ['subject_id', 'gender', 'anchor_age', 'anchor_year'])
    cohort = admissions.merge(dx, on=KEY, validate='one_to_one')
    cohort = cohort.merge(patients, on='subject_id', validate='many_to_one')
    counts = {'hf_admissions_matched': len(cohort)}
    cohort['age_years'] = cohort.anchor_age + cohort.admittime.dt.year - cohort.anchor_year
    cohort = cohort.loc[cohort.age_years.ge(18)].copy()
    counts['adult_hf_admissions'] = len(cohort)
    cohort['cutoff'] = cohort.admittime + pd.Timedelta(hours=24)
    cohort = cohort.loc[cohort.dischtime.gt(cohort.cutoff)].copy()
    counts['adult_stays_past_landmark'] = len(cohort)
    died = cohort.hospital_expire_flag.eq(1)
    valid_death = cohort.deathtime.gt(cohort.cutoff) & cohort.deathtime.le(cohort.dischtime)
    valid_survivor = cohort.hospital_expire_flag.eq(0) & (
        cohort.deathtime.isna() | cohort.deathtime.gt(cohort.dischtime))
    cohort = cohort.loc[(died & valid_death) | valid_survivor].copy()
    counts['eligible_admissions'] = len(cohort)
    cohort = cohort.sort_values(['subject_id', 'admittime', 'hadm_id']).drop_duplicates('subject_id')
    if cohort.empty:
        raise ValueError('No eligible admissions')
    counts['index_patients'] = len(cohort)
    cohort[LABEL] = cohort.hospital_expire_flag.astype(int)
    cohort['age_years'] = cohort.age_years.clip(upper=89).astype(int)
    cohort['split'] = cohort.subject_id.map(assign_split)
    return cohort, counts


def extract_icu(root, cohort):
    stays = read(root, 'icu/icustays', KEY + ['stay_id', 'intime', 'outtime'],
                 parse_dates=['intime', 'outtime'])
    stays = stays.merge(cohort[KEY + ['admittime', 'cutoff']], on=KEY, validate='many_to_one')
    stays['start'] = stays[['admittime', 'intime']].max(axis=1, skipna=False)
    stays = stays.loc[stays.intime.lt(stays.cutoff) & stays.outtime.gt(stays.start)].copy()
    stays['hours'] = (stays[['outtime', 'cutoff']].min(axis=1) - stays.start).dt.total_seconds() / 3600
    counts = stays.groupby('hadm_id').agg(icu_stays_24h=('stay_id', 'size'), icu_hours_24h=('hours', 'sum'))
    cohort = cohort.join(counts, on='hadm_id')
    cohort['icu_stays_24h'] = cohort.icu_stays_24h.fillna(0).astype(int)
    cohort['icu_hours_24h'] = cohort.icu_hours_24h.fillna(0).clip(upper=24)
    return cohort, stays


def summarize(root, table, items, dictionary, index, chunksize):
    is_icu = table.startswith('icu/')
    keys = KEY + (['stay_id'] if is_icu else [])
    fields = keys + ['itemid', 'charttime', 'storetime', 'valuenum', 'valueuom']
    labels = read(root, dictionary, ['itemid', 'label']).set_index('itemid').label
    missing_ids = set(items) - set(labels.index)
    if missing_ids:
        raise ValueError(f'Missing item IDs in {dictionary}: {sorted(missing_ids)}')
    print(f'Scanning {table}', flush=True)
    summaries = []
    for number, chunk in enumerate(read(root, table, fields, chunksize=chunksize), 1):
        if number % 50 == 0:
            print(f'{table}: scanned {number * chunksize:,} source rows', flush=True)
        chunk = chunk.loc[chunk.itemid.isin(items) & chunk.hadm_id.isin(index.hadm_id)]
        if chunk.empty:
            continue
        chunk = chunk.merge(index, on=keys, validate='many_to_one')
        chart = pd.to_datetime(chunk.charttime, errors='coerce')
        store = pd.to_datetime(chunk.storetime, errors='coerce')
        eligible = chart.ge(chunk.start) & chart.lt(chunk.cutoff) & store.le(chunk.cutoff)
        if is_icu:
            eligible &= chart.lt(chunk.outtime)
        chunk = chunk.loc[eligible].copy()
        chunk['valuenum'] = pd.to_numeric(chunk.valuenum, errors='coerce')
        chunk['valueuom'] = chunk.valueuom.astype('string').str.lower().str.strip()
        for item_id, (name, units, low, high) in items.items():
            selected = chunk.itemid.eq(item_id) & chunk.valuenum.gt(low) & chunk.valuenum.le(high)
            if units:
                selected &= chunk.valueuom.isin(units)
            values = chunk.loc[selected, ['hadm_id', 'valuenum']].copy()
            if values.empty:
                continue
            if item_id == 223761:
                values['valuenum'] = (values.valuenum - 32) / 1.8
            summary = values.groupby('hadm_id').valuenum.agg(['count', 'sum', 'min', 'max']).reset_index()
            summaries.append(summary.assign(feature=name))
    if summaries:
        result = pd.concat(summaries).groupby(['hadm_id', 'feature']).agg(
            {'count': 'sum', 'sum': 'sum', 'min': 'min', 'max': 'max'})
        result['mean'] = result['sum'] / result['count']
        result = result.drop(columns='sum').unstack('feature')
        result.columns = [f'{feature}_{stat}' for stat, feature in result.columns]
    else:
        result = pd.DataFrame(index=pd.Index([], name='hadm_id'))
    for name, *_ in items.values():
        for stat in ('count', 'mean', 'min', 'max'):
            column = f'{name}_{stat}'
            if column not in result:
                result[column] = np.nan
    return result, {str(item): str(labels.loc[item]) for item in items}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New directory outside the source tree')
    parser.add_argument('--mini-size', type=int, default=1000)
    parser.add_argument('--chunksize', type=int, default=750_000)
    args = parser.parse_args()
    if min(args.mini_size, args.chunksize) < 1:
        parser.error('mini-size and chunksize must be positive')
    root, out = args.root.resolve(), args.out.resolve()
    if out.is_relative_to(root) or out.exists():
        parser.error('--out must be a new directory outside --root')
    cohort, counts = extract_index(root)
    print('Index cohort selected', flush=True)
    lab_index = cohort[KEY + ['admittime', 'cutoff']].rename(columns={'admittime': 'start'})
    labs, lab_labels = summarize(root, 'hosp/labevents', LABS, 'hosp/d_labitems', lab_index, args.chunksize)
    cohort = cohort.join(labs, on='hadm_id')
    icu_tables = ['icu/icustays', 'icu/chartevents', 'icu/d_items']
    has_stays = (root / 'icu/icustays.csv.gz').is_file()
    has_vitals = all((root / f'{table}.csv.gz').is_file() for table in icu_tables)
    vital_labels = {}
    if has_stays:
        cohort, stays = extract_icu(root, cohort)
        if has_vitals:
            vitals, vital_labels = summarize(root, 'icu/chartevents', VITALS, 'icu/d_items',
                                            stays[KEY + ['stay_id', 'start', 'cutoff', 'outtime']], args.chunksize)
            cohort = cohort.join(vitals, on='hadm_id')
    for column in cohort.filter(regex='_count$'):
        cohort[column] = cohort[column].fillna(0).astype(int)
    cohort = cohort.drop(columns=KEY + ['admittime', 'dischtime', 'deathtime', 'cutoff',
                                        'anchor_age', 'anchor_year', 'hospital_expire_flag'])
    files = ['hosp/admissions', 'hosp/patients', 'hosp/diagnoses_icd', 'hosp/d_labitems', 'hosp/labevents']
    files += icu_tables if has_vitals else ['icu/icustays'] if has_stays else []
    manifest = {
        'release_note': (root / 'CHANGELOG.txt').read_text(encoding='utf-8').splitlines()[0]
        if (root / 'CHANGELOG.txt').exists() else None,
        'files_bytes': {f'{name}.csv.gz': (root / f'{name}.csv.gz').stat().st_size for name in files},
        'counts': counts, 'seed': SEED, 'python': sys.version.split()[0],
        'pandas': pd.__version__, 'numpy': np.__version__, 'mini_size': args.mini_size,
        'lab_item_labels': lab_labels, 'vital_item_labels': vital_labels,
        'nonmissing': cohort.count().to_dict(),
    }
    out.mkdir(parents=True, exist_ok=False)
    cohort.to_csv(out / 'cohort.csv', index=False)
    cohort.sample(n=min(len(cohort), args.mini_size), random_state=SEED).to_csv(out / 'cohort_mini.csv', index=False)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('Cohort files and manifest written', flush=True)


if __name__ == '__main__':
    main()
