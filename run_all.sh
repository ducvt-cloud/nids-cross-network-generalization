#!/usr/bin/env bash
# Run all experiments in order. Intended for a Colab-style environment with the
# NetFlow-v2 / DAPT2020 data mounted under Google Drive (see README).
# Each script is self-contained and can also be run on its own.
set -e
cd "$(dirname "$0")"
for f in src/01_indomain_leakage_check.py \
         src/02_transfer_matrix.py \
         src/03_domain_adaptation_coral.py \
         src/04_dann_and_fewshot.py \
         src/05_adaptation_5seed_std.py \
         src/06_per_attack_type.py \
         src/07_fullscale_confirmation.py; do
  echo "==================================================================="
  echo "RUNNING: $f"
  echo "==================================================================="
  python "$f"
done
echo "All experiments finished."
