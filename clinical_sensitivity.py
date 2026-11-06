"""Clinical sensitivity analyses for the client-level calibration study.

Runs locally on credentialed MIMIC-IV v3.1 and writes aggregate results only.
Counts below 10 are suppressed in descriptive tables.

Usage (project root, next to client_calibration_study.py):

    python clinical_sensitivity.py --root "<mimic-iv-3.1 folder>" --out local_output/clinical_sensitivity

Analyses
  D1  Principal-diagnosis share and mortality by service and era.
  D2  Discharge to hospice among survivors, by service and era.
  D3  Admissions coded with hypertensive heart failure combination codes but no 428/I50 code.
  D1b Principal-diagnosis share with the broad definition used in S3b.
  D4  Baseline characteristics by era (medians and interquartile ranges, proportions).
  D5  Index admissions coded with COVID-19 (U07.1) by era.
  S1  Main analysis with oncology (OMED) and neurology (NMED) grouped with general medicine.
  S2  Main analysis with the composite outcome death after the landmark or discharge to hospice.
  S3  Main analysis restricted to patients with heart failure as the principal diagnosis.
  S3b As S3, also counting a principal hypertensive heart failure code (I11.0, I13.0, I13.2, 402.x1, 404.x1/x3).
  S4  Main analysis excluding index admissions coded with COVID-19 (U07.1).
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_cohort import LABEL, extract_index, read
import client_calibration_study as ccs

SMALL = 10
EXTRA_HF_ICD10 = ('I110', 'I130', 'I132')
EXTRA_HF_ICD9 = ('40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493')
ALT_GROUP = dict(ccs.SERVICE_GROUP, OMED='General medicine', NMED='General medicine')


def suppress(n):
    return int(n) if n >= SMALL else f'<{SMALL}'


def describe(table, by, value):
    """Counts and rates by group, with small-cell suppression."""
    out = {}
    for key, part in table.groupby(by):
        key = ' | '.join(map(str, key)) if isinstance(key, tuple) else str(key)
        n = len(part)
        k = int(part[value].sum())
        out[key] = dict(patients=suppress(n), count=suppress(k),
                        proportion=round(k / n, 4) if n >= SMALL and k >= SMALL else None)
    return out


def summary_only(result):
    """Keep the parts of an analysis needed for the manuscript."""
    keep = ('scheme', 'clients', 'counts', 'optimisation', 'update_parameters', 'fallbacks_full_data',
            'fallback_counts_bootstrap', 'seconds')
    slim = {k: result[k] for k in keep}
    slim['point'] = {m: dict(summary=row['summary'],
                             overall={k: v for k, v in row['overall'].items() if k != 'bins'},
                             clients={c: {k: v for k, v in r.items() if k != 'bins'} for c, r in row['clients'].items()})
                     for m, row in result['point'].items()}
    slim['intervals'] = result['intervals']
    slim['paired'] = result['paired']
    return slim


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--cohort', type=Path, default=Path('local_output/audited_cohort/cohort.csv'))
    parser.add_argument('--config', type=Path, default=Path('local_output/experiments.json'))
    parser.add_argument('--cohort-sha256', default=ccs.COHORT_HASH)
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
    split = ccs.temporal_split(args.root, index, 2017, 2020).to_numpy()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    raw_service = ccs.landmark_service(args.root, index)
    service = np.array([ccs.SERVICE_GROUP.get(s, ccs.OTHER) for s in raw_service], dtype=object)

    admissions = read(args.root, 'hosp/admissions', ['hadm_id', 'discharge_location'])
    location = index.hadm_id.map(admissions.set_index('hadm_id').discharge_location)
    hospice = location.fillna('').str.upper().str.contains('HOSPICE').to_numpy()
    death = frame[LABEL].to_numpy(dtype=int)
    principal = frame.hf_principal.astype(bool).to_numpy()

    era = pd.Series(split).map({'train': 'Development', 'validation': 'Updating', 'test': 'Test'}).to_numpy()
    used = split != 'excluded'
    table = pd.DataFrame(dict(service=service, era=era, death=death, principal=principal,
                              hospice_survivor=hospice & (death == 0), survivor=death == 0))[used]
    report = dict(cohort_sha256=args.cohort_sha256, bootstrap_draws=args.bootstrap, small_cell=SMALL)
    report['D1_principal_share'] = describe(table, ['service', 'era'], 'principal')
    report['D1_mortality_principal'] = describe(table[table.principal], ['service', 'era'], 'death')
    report['D1_mortality_secondary'] = describe(table[~table.principal], ['service', 'era'], 'death')
    report['D2_hospice_among_survivors'] = describe(table[table.survivor], ['service', 'era'], 'hospice_survivor')
    report['D2_hospice_overall_survivors'] = describe(table[table.survivor].assign(all='All'), ['all'], 'hospice_survivor')

    # D3: admissions with hypertensive HF combination codes and no 428/I50 code (admission level).
    dx = read(args.root, 'hosp/diagnoses_icd', ['hadm_id', 'seq_num', 'icd_version', 'icd_code'], dtype={'icd_code': 'string'})
    code = dx.icd_code.str.strip().str.upper().str.replace('.', '', regex=False)
    core = (dx.icd_version.eq(9) & code.str.startswith('428')) | (dx.icd_version.eq(10) & code.str.startswith('I50'))
    extra = (dx.icd_version.eq(10) & code.isin(EXTRA_HF_ICD10)) | (dx.icd_version.eq(9) & code.isin(EXTRA_HF_ICD9))
    covid_hadm = set(dx.hadm_id[dx.icd_version.eq(10) & code.eq('U071')])
    # Coding rules sequence I11.0/I13.x (402.x1/404.x1/x3) before the I50/428 code, so a principal
    # hypertensive heart failure code also marks heart failure as the reason for admission.
    broad_hadm = set(dx.hadm_id[pd.to_numeric(dx.seq_num, errors='coerce').eq(1) & (core | extra).fillna(False)])
    flags = pd.DataFrame(dict(hadm_id=dx.hadm_id, core=core.fillna(False), extra=extra.fillna(False)))
    flags = flags.groupby('hadm_id')[['core', 'extra']].max()
    report['D3_admissions'] = dict(core_codes=suppress(int(flags.core.sum())),
                                   extra_codes_only=suppress(int((flags.extra & ~flags.core).sum())),
                                   extra_codes_with_core=suppress(int((flags.extra & flags.core).sum())),
                                   note='Admission-level counts over all MIMIC-IV hospital admissions, before age, '
                                        'landmark and first-admission criteria.')

    principal_broad = principal | index.hadm_id.isin(broad_hadm).to_numpy()
    table['principal_broad'] = principal_broad[used]
    report['D1b_principal_broad_share'] = describe(table, ['service', 'era'], 'principal_broad')
    report['D1b_mortality_secondary_broad'] = describe(table[~table.principal_broad], ['service', 'era'], 'death')

    # D4: baseline characteristics by era.
    base = frame.assign(era=era)[used]
    def med(col):
        return {e: (None if part[col].notna().sum() < SMALL else
                    [round(float(v), 1) for v in part[col].quantile([0.5, 0.25, 0.75])])
                for e, part in base.groupby('era')}
    d4 = {'patients': {e: suppress(len(p)) for e, p in base.groupby('era')},
          'age_years_median_iqr': med('age_years'),
          'female_proportion': {e: round(float(p.gender.eq('F').mean()), 3) for e, p in base.groupby('era')},
          'early_icu_proportion': {e: round(float(p.icu_stays_24h.gt(0).mean()), 3) for e, p in base.groupby('era')},
          'principal_hf_proportion': {e: round(float(p.hf_principal.astype(bool).mean()), 3) for e, p in base.groupby('era')},
          'in_hospital_death_proportion': {e: round(float(p[LABEL].mean()), 4) for e, p in base.groupby('era')}}
    for col in ('bun_mg_dl_mean', 'creatinine_mg_dl_mean', 'sodium_meq_l_mean', 'hemoglobin_g_dl_mean',
                'albumin_g_dl_mean', 'noninvasive_sbp_mmhg_mean', 'heart_rate_bpm_mean', 'spo2_pct_mean'):
        if col in base.columns:
            d4[f'{col}_median_iqr'] = med(col)
            d4[f'{col}_missing_proportion'] = {e: round(float(p[col].isna().mean()), 3) for e, p in base.groupby('era')}
    report['D4_baseline_by_era'] = d4

    covid = index.hadm_id.isin(covid_hadm).to_numpy()
    report['D5_covid_index_admissions'] = {e: suppress(int((covid & (era == e)).sum())) for e in ('Development', 'Updating', 'Test')}

    rng = np.random.default_rng(ccs.SEED + 7)
    analyses = {}

    def attempt(name, groups, labels, data, era_split):
        print(name, flush=True)
        try:
            merged, log = ccs.merge_small(groups, labels, era_split, ccs.service_fallback)
            if np.unique(merged[era_split != 'excluded']).size < 2:
                analyses[name] = dict(failed='Fewer than two clients met 30 events and 30 non-events per era', merges=log)
                return
            analyses[name] = dict(merges=log, **summary_only(ccs.analyse(name, merged, data, era_split, config, args, rng)))
        except ValueError as error:  # e.g. too few events for any valid client set
            analyses[name] = dict(failed=str(error))

    alt = np.array([ALT_GROUP.get(s, ccs.OTHER) for s in raw_service], dtype=object)
    attempt('S1_alternative_grouping', alt, death, frame, split)

    composite = frame.copy()
    composite[LABEL] = ((death == 1) | hospice).astype(int)
    attempt('S2_death_or_hospice', service.copy(), composite[LABEL].to_numpy(), composite, split)

    keep = principal
    attempt('S3_principal_hf_only', service[keep].copy(), death[keep], frame.loc[keep].reset_index(drop=True), split[keep])

    keep = principal_broad
    attempt('S3b_principal_hf_broad', service[keep].copy(), death[keep], frame.loc[keep].reset_index(drop=True), split[keep])

    keep = ~covid
    attempt('S4_excluding_covid', service[keep].copy(), death[keep], frame.loc[keep].reset_index(drop=True), split[keep])

    report['analyses'] = analyses
    args.out.mkdir(parents=True)
    (args.out / 'clinical_sensitivity_results.json').write_text(
        json.dumps(report, indent=2, allow_nan=False, default=float), encoding='utf-8')
    print(f'Wrote aggregate results to {args.out}')


if __name__ == '__main__':
    main()
