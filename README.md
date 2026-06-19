# Rigorous Evaluation of Deep Learning Generalization for APT / Network Intrusion Detection

Reproducibility package for the paper **"Rigorous Evaluation of Deep Learning Generalization for APT / Network Intrusion Detection: From In-Domain Illusion to Few-Shot Adaptation."**

This repository contains the complete, self-contained code that reproduces every table in the paper: the leakage-free evaluation protocol, the cross-network transfer matrix, the unsupervised-adaptation (CORAL / DANN) and few-shot experiments, the per-attack-type decomposition, and the large-scale confirmation.

---

## What the paper shows

Under a strict, leakage-free protocol on four NetFlow-v2 datasets and three model families, (1) in-domain AUROC saturates near 0.99 but does **not** predict cross-network performance; (2) cross-network transfer **collapses and frequently inverts** (AUROC < 0.5), model-agnostically; (3) the cause is a **label-conditional inversion** of the feature–label relationship driven by differing attack-mix composition; (4) unsupervised domain adaptation (CORAL, DANN) is insufficient on the inverted pairs, whereas a few dozen labeled target examples per class restore both AUROC and AUPRC to near the in-domain ceiling.

---

## Datasets (not included — third-party, public)

No data is redistributed here. Obtain the datasets from their original providers:

- **NetFlow-v2 family** (NF-CSE-CIC-IDS2018-v2, NF-UNSW-NB15-v2, NF-BoT-IoT-v2, NF-ToN-IoT-v2): the standardized NetFlow-v2 datasets released by the University of Queensland network-security dataset collection. All four share a common NetFlow-v2 feature schema; the experiments use the intersection of numeric features common to the datasets being compared.
- **DAPT2020** (multi-stage APT case study): obtain from its original authors.

### Expected layout

The scripts are written for Google Colab and read from Google Drive. Default paths:

```
/content/drive/MyDrive/APT_Data/
    NF-UNSW-NB15-v2/NF-UNSW-NB15-v2.csv
    NF-CSE-CIC-IDS2018-v2.zip
    NF-BoT-IoT-v2.zip
    NF-ToN-IoT-v2.rar
/content/drive/MyDrive/DAPT2020/        (for the temporal case study)
```

To run elsewhere, edit the `DRIVE` / `SRC` path variables at the top of each script and remove the `from google.colab import drive; drive.mount(...)` lines. Network identifiers (IP addresses, ports, flow IDs, timestamps as features) are dropped via the `ID_DROP` / `DROP_COLS` list so they cannot leak.

---

## Environment

- Python 3.10+ (developed on Google Colab, single GPU; CPU also works but is slower).
- Install dependencies: `pip install -r requirements.txt`
- The `.rar` ToN-IoT archive is unpacked automatically (the scripts try `unrar`, then `p7zip`, then `patool`). If ToN cannot be unpacked, the matrix scripts drop it and continue with the remaining datasets.

---

## Reproducibility settings

- Global seed: `SEED = 42`. Multi-seed runs use seeds `42 .. 42+N_SEEDS-1`.
- Per-script knobs at the top of each file: `MAX_ROWS_PER_DS` (uniform subsample size), `N_SEEDS`, epochs, batch size. The paper's main tables use a 150,000-flow subsample (`MAX_ROWS_PER_DS = 150_000`); the large-scale confirmation uses ~1,000,000.
- Splitting is leakage-free: training-only normalization and threshold selection; entity-grouped (source-host) splitting for the in-domain leakage check; temporal ordering where timestamps exist; multi-seed repetition.
- Primary metric is **AUROC** (base-rate invariant). AUPRC is secondary and must be read against each target's base rate. The BoT target has a 99.6% attack base rate, so its AUPRC is a base-rate artifact and is excluded from interpretation.

---

## Scripts → paper tables

| Script | Reproduces | Notes |
|--------|-----------|-------|
| `src/01_indomain_leakage_check.py` | Table 2 (in-domain: random vs entity-grouped split) | RF/LGBM/MLP; confirms in-domain ~0.99 is genuine, not host memorization |
| `src/02_transfer_matrix.py` | Tables 3 & 4 (cross-domain AUROC / AUPRC matrix) | RF + LGBM + MLP, 5 seeds, mean±std, with target base rates |
| `src/03_domain_adaptation_coral.py` | §5.4 (Deep CORAL) | baseline vs CORAL per source→target pair |
| `src/04_dann_and_fewshot.py` | §5.4 (DANN + few-shot) | DANN vs baseline on inverted pairs; few-shot AUROC/AUPRC by k |
| `src/05_adaptation_5seed_std.py` | Table 6 (few-shot) + §5.4 numbers | CORAL/DANN and few-shot, AUROC/AUPRC mean±std over 5 seeds, k={0,10,50,100}+ceiling |
| `src/06_per_attack_type.py` | Table 8 / Appendix C (per-attack-type) | one-vs-benign AUROC per target attack family; localizes the inversion |
| `src/07_fullscale_confirmation.py` | Table 7 / Appendix B (large-scale) | 4×4 AUROC at ~1,000,000 flows/dataset, LightGBM, 3 seeds |

Each script prints its results to stdout and is independently runnable.

---

## Running

Each file is self-contained (it reloads and re-aligns the data itself), so you can run any single experiment in isolation, e.g. in a Colab cell or:

```bash
python src/02_transfer_matrix.py
```

`run_all.sh` runs them in order (intended for a Colab-style environment with the data mounted).

---

## Citation

If you use this code, please cite the paper (full reference to be added on acceptance):

> Thanh Duc Vu, Cho Do Xuan, Long Giang Nguyen. *Rigorous Evaluation of Deep Learning Generalization for APT / Network Intrusion Detection: From In-Domain Illusion to Few-Shot Adaptation.*

## License

Code released under the MIT License (see `LICENSE`). The datasets are the property of their respective providers and are not covered by this license.
