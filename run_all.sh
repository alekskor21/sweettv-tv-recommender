#!/bin/sh
# Re-run the whole pipeline from the raw CSVs in data/ and keep each stage's console output.
# Usage (from the repository root): sh run_all.sh
set -e
PY=.venv/bin/python
mkdir -p outputs/logs
for s in 00_prepare 01_eda 02_time_alignment 02b_dst_check 03_join 03b_verify_join 04_threshold_eda \
         05_repeat_stats check_metric 10_baselines 11_gt_sensitivity 12_baseline_submissions 13_gt_grid \
         14_count_hypothesis 15_viewcount_variant 20_als 21_mf_variants 30_content 31_content_ablation \
         40_methods_eval 50_hybrid_eval 51_hybrid_experiments 52_robustness 60_final_submission 70_examples; do
  echo "== $s"
  if [ "$s" = "60_final_submission" ]; then
    $PY src/$s.py s03_hybrid 2>&1 | grep -v Warning | tee outputs/logs/$s.log
  else
    $PY src/$s.py 2>&1 | grep -v Warning | tee outputs/logs/$s.log
  fi
done
shasum -a 256 outputs/submissions/*.csv
