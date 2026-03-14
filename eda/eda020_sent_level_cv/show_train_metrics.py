"""metrics_log.jsonからlast epochの学習時eval metricsを表示"""
import json

print(f"{'exp':>6} {'fold':>5} | {'train_loss':>10} {'eval_loss':>10} {'eval_chrf':>10} {'eval_bleu':>10} {'eval_geo':>10}")
print("-" * 80)

for exp, exp_dir in [('exp023', 'exp023_full_preprocessing'), ('exp019', 'exp019_sent_additional')]:
    for fold in range(5):
        mlog = f'/home/user/work/Akkadian/workspace/{exp_dir}/results/fold{fold}/metrics_log.json'
        try:
            with open(mlog) as f:
                data = json.load(f)
            last_eval = data['eval'][-1]
            last_train = data['train'][-1] if data['train'] else {}
            print(f"{exp:>6} fold{fold} | {last_train.get('loss', 0):10.4f} {last_eval.get('eval_loss', 0):10.4f} {last_eval.get('eval_chrf', 0):10.2f} {last_eval.get('eval_bleu', 0):10.2f} {last_eval.get('eval_geo_mean', 0):10.2f}")
        except Exception as e:
            print(f"{exp:>6} fold{fold} | ERROR: {e}")
    print("-" * 80)
