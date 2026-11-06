# Client-level calibration of a federated heart failure mortality model

Code for the manuscript:

> Dawod AA, Qazza A, Qazza R. *Client-Level Calibration of a Federated Mortality Prediction Model under Temporal Shift: Local, Federated and Personalized Updating in Heart Failure.* (submitted)

The code simulates federated learning across hospital services in MIMIC-IV version 3.1. It then compares four ways of updating the federated model on recent data: federated recalibration, local recalibration, personalized intercepts, and updates estimated with federated Newton iterations. Each strategy is evaluated by calibration within clients and by net benefit at fixed risk thresholds.

**No data are included.** MIMIC-IV is available only to credentialed users from PhysioNet (<https://physionet.org/content/mimiciv/3.1/>) under the PhysioNet Credentialed Health Data Use Agreement. Only credentialed users may run the scripts on their own copy of the data. Do not upload MIMIC-IV data or derived patient-level files to this repository or to online services.

## Requirements

Python ≥ 3.10 with NumPy, pandas and Matplotlib (see `requirements.txt`). The analyses in the paper were run on Windows with the versions recorded by `environment_report.py`; they are listed in the manuscript.

```
python -m pip install -r requirements.txt
```

## Files

| File | Purpose |
|---|---|
| `build_cohort.py` | Builds the 24-hour landmark heart failure cohort from the MIMIC-IV CSV files |
| `run_experiments.py`, `q1_experiments.py` | Shared feature construction and the FedAvg optimizer used by the audited cohort |
| `calib_core.py` | Calibration metrics, recalibration estimators, federated Newton iterations and net benefit |
| `client_calibration_study.py` | Main analysis: clients, temporal eras, updating strategies, bootstrap and learning curves |
| `clinical_sensitivity.py` | Clinical descriptive tables and sensitivity analyses S1–S4 |
| `test_calib_core.py` | Unit tests of the calibration core |
| `environment_report.py` | Records the Python, package and processor versions |
| `config/experiments.json` | Optimization settings (seed, rounds, local steps, learning rate) |
| `manuscript_tools/` | Figures and tables from the aggregate results JSON only |

## Reproducing the analysis

Replace `<MIMIC>` with the folder that contains `hosp/` and `icu/` of MIMIC-IV 3.1.

```
python -m unittest test_calib_core.py
python build_cohort.py --root <MIMIC> --out local_output/audited_cohort
python client_calibration_study.py --root <MIMIC> --out local_output/client_calibration --config config/experiments.json
python clinical_sensitivity.py --root <MIMIC> --out local_output/clinical_sensitivity --config config/experiments.json
python environment_report.py
python manuscript_tools/make_figures.py local_output/client_calibration/client_calibration_results.json figures
```

Before every analysis, the scripts check the SHA-256 fingerprint of the cohort file against the value used in the paper (`8ec032ce…992d`); a different MIMIC-IV version or cohort definition stops the run. The analysis writes only aggregate results: no identifiers, dates, individual predictions or model weights. Calibration bins with fewer than 30 patients and descriptive counts below 10 are suppressed.

## Privacy note

The clients are arrays inside one process. The code is a simulation of federated estimation; it provides no transport security, secure aggregation or formal privacy guarantee.

## Licence and citation

Code: MIT License (see `LICENSE`). Please cite the article and MIMIC-IV (Johnson et al., *Scientific Data* 2023; PhysioNet, doi:10.13026/kpb9-mt58).
