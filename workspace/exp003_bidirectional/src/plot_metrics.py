"""
学習曲線可視化ツール
- 単一実験: train loss, eval metrics vs epoch
- 複数実験: 比較プロット

Usage:
  python plot_metrics.py                          # この実験のみ
  python plot_metrics.py --compare exp002 exp003  # 複数実験比較
"""
import os
import sys
import json
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
WORKSPACE_DIR = os.path.dirname(EXP_DIR)


def load_metrics(exp_dir):
    """metrics_log.json を読み込む"""
    path = os.path.join(exp_dir, "results", "metrics_log.json")
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def plot_single(exp_dir, save_dir=None):
    """単一実験の学習曲線"""
    data = load_metrics(exp_dir)
    if data is None:
        return

    exp_name = os.path.basename(exp_dir)
    save_dir = save_dir or os.path.join(exp_dir, "results")

    train = data["train"]
    eval_data = data["eval"]

    # Train loss
    train_epochs = [d["epoch"] for d in train if "epoch" in d]
    train_loss = [d["loss"] for d in train if "loss" in d]

    # Eval metrics
    eval_epochs = [d["epoch"] for d in eval_data if "epoch" in d]
    eval_loss = [d["eval_loss"] for d in eval_data if "eval_loss" in d]
    eval_chrf = [d.get("eval_chrf", 0) for d in eval_data]
    eval_bleu = [d.get("eval_bleu", 0) for d in eval_data]
    eval_geo = [d.get("eval_geo_mean", 0) for d in eval_data]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{exp_name} Training Curves", fontsize=14)

    # Loss
    axes[0].plot(train_epochs, train_loss, "b-", alpha=0.6, label="Train Loss")
    axes[0].plot(eval_epochs, eval_loss, "r-o", markersize=4, label="Eval Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Eval Metrics
    axes[1].plot(eval_epochs, eval_chrf, "g-o", markersize=4, label="chrF")
    axes[1].plot(eval_epochs, eval_bleu, "b-o", markersize=4, label="BLEU")
    axes[1].plot(eval_epochs, eval_geo, "r-o", markersize=4, label="geo_mean")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Evaluation Metrics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Geo mean zoomed
    axes[2].plot(eval_epochs, eval_geo, "r-o", markersize=5, linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("geo_mean")
    axes[2].set_title("geo_mean (Primary Metric)")
    axes[2].grid(True, alpha=0.3)
    if eval_geo:
        best_idx = max(range(len(eval_geo)), key=lambda i: eval_geo[i])
        axes[2].annotate(
            f"Best: {eval_geo[best_idx]:.2f} (ep{eval_epochs[best_idx]:.0f})",
            xy=(eval_epochs[best_idx], eval_geo[best_idx]),
            xytext=(10, -20), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=10, color="red",
        )

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_comparison(exp_names, save_dir=None):
    """複数実験の比較プロット"""
    save_dir = save_dir or os.path.join(WORKSPACE_DIR)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Experiment Comparison", fontsize=14)

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    for i, name in enumerate(exp_names):
        exp_dir = os.path.join(WORKSPACE_DIR, name)
        data = load_metrics(exp_dir)
        if data is None:
            continue

        eval_data = data["eval"]
        eval_epochs = [d["epoch"] for d in eval_data if "epoch" in d]
        eval_geo = [d.get("eval_geo_mean", 0) for d in eval_data]
        eval_loss = [d["eval_loss"] for d in eval_data if "eval_loss" in d]

        color = colors[i % len(colors)]
        axes[0].plot(eval_epochs, eval_loss, "-o", color=color, markersize=3, label=name)
        axes[1].plot(eval_epochs, eval_geo, "-o", color=color, markersize=3, label=name)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Eval Loss")
    axes[0].set_title("Eval Loss Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("geo_mean")
    axes[1].set_title("geo_mean Comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "experiment_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", nargs="+", help="Experiment names to compare")
    args = parser.parse_args()

    if args.compare:
        plot_comparison(args.compare)
    else:
        plot_single(EXP_DIR)


if __name__ == "__main__":
    main()
