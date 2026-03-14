#!/bin/bash
set -e
cd /home/user/work/Akkadian

echo "=== exp023 GKF fold0 re-eval (with CSV save) ==="
python eda/eda020_sent_level_cv/eval_full_doc.py \
    workspace/exp023_full_preprocessing/results/fold0/last_model \
    exp023_gkf_fold0_last \
    --preprocess exp023 --fold 0

echo "=== exp019 GKF fold0 re-eval (with CSV save) ==="
python eda/eda020_sent_level_cv/eval_full_doc.py \
    workspace/exp019_sent_additional/results/fold0/last_model \
    exp019_gkf_fold0_last \
    --fold 0

echo "All done"
