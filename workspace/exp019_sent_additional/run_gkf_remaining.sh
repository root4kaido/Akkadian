#!/bin/bash
set -e
cd /home/user/work/Akkadian

for FOLD in 1 2 3 4; do
    echo "============================================"
    echo "=== exp019 GKF fold${FOLD} Training ==="
    echo "============================================"
    python workspace/exp019_sent_additional/src/train_gkf.py --fold ${FOLD}

    echo "=== exp019 GKF fold${FOLD} Evaluation ==="
    python eda/eda020_sent_level_cv/eval_full_doc.py \
        workspace/exp019_sent_additional/results/fold${FOLD}/last_model \
        exp019_gkf_fold${FOLD}_last \
        --fold ${FOLD}

    echo "=== exp019 GKF fold${FOLD} DONE ==="
done

echo "All exp019 GKF folds complete!"
