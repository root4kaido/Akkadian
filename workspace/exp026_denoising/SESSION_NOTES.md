# SESSION_NOTES: exp026_denoising

## セッション情報
- **日付**: 2026-03-13
- **作業フォルダ**: workspace/exp026_denoising
- **目標**: Denoising Augmentationでドメイン間の構文揺れへの頑健性を向上

## 仮説
GKFでドメインが完全に未見になると、表層的なパターンマッチングが通用しない。
ソース側（翻字）にノイズを注入して学習することで、
モデルが文脈ベースの翻訳を学び、構文の揺れに強くなる。
ByT5のspan corruption事前学習と相性が良いはず。

## 手法
- 文字レベルのノイズ: drop(33%) / swap(33%) / insert(33%)
- 各文字に確率 noise_prob=0.1 で適用
- forward方向(Akk→Eng)の50%のサンプルにのみ適用
- backward方向(Eng→Akk)にはノイズなし
- valデータはクリーンのまま

## 試したアプローチと結果

| アプローチ | 変更点 | sent-CV (geo) | doc-CV (geo) | 備考 |
|-----------|--------|---------------|--------------|------|
| exp023 GKF fold3 (ベースライン) | - | 35.44 | 25.52 | - |
| exp026 Denoising p=0.1 fold3 | ソースノイズ | **33.55** | **23.21** | **悪化**。sent -1.89pt, doc -2.31pt |

## ファイル構成
- `src/train_gkf.py` — Denoising版学習スクリプト

## 重要な知見
- **Denoising Augmentation（文字ノイズ）は逆効果**。sent-geo -1.89pt、doc-geo -2.31pt悪化
- アッカド語翻字はバイトレベルで意味が変わるため、文字ノイズがモデルを混乱させる
- ByT5のspan corruption事前学習との相性は期待ほどではなかった
- rep=11.4%(sent)は低いが、スコア自体が下がっているので有意義とは言えない

## コマンド履歴
```bash
# fold0実行
python workspace/exp026_denoising/src/train_gkf.py --fold 0 --noise_prob 0.1

# 評価
python eda/eda020_sent_level_cv/eval_full_doc.py \
    workspace/exp026_denoising/results/fold0/last_model \
    exp026_denoising_fold0_last --preprocess exp023 --fold 0
```

## 次のステップ
- **棄却**。文字レベルノイズはアッカド語翻字には不適切。他のaugment手法を検討すべき
