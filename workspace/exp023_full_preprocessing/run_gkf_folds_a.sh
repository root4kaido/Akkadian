#!/bin/bash
# Folds 1, 3 (dev0用)
set -e
cd /home/user/work/Akkadian

for FOLD in 1 3; do
    echo "=== Training fold ${FOLD} ==="
    python workspace/exp023_full_preprocessing/src/train_gkf.py --fold ${FOLD}
    echo "=== Eval fold ${FOLD} ==="
    python eda/eda020_sent_level_cv/eval_full_doc.py \
        workspace/exp023_full_preprocessing/results/fold${FOLD}/last_model \
        exp023_gkf_fold${FOLD}_last \
        --preprocess exp023 --fold ${FOLD}
    echo "=== Fold ${FOLD} done ==="
done
echo "All done (folds 1,3)"
