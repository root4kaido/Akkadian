"""
OA_Lexicon_eBL.csvからform→typeの辞書を構築しJSONで保存する。
曖昧なform（複数typeに該当）はambiguous_strategy設定に従い処理。

Usage:
    python build_lexicon.py

Output:
    dataset/form_type_dict.json — {"form": "PN"|"GN"|null, ...}
"""
import json
import os
import sys

import pandas as pd
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)

with open(os.path.join(EXP_DIR, "config.yaml")) as f:
    config = yaml.safe_load(f)


def build_form_type_dict(lexicon_path: str, tag_types: list, ambiguous_strategy: str) -> dict:
    """
    OA_Lexiconからform→tag辞書を構築。

    Returns: {form: "PN"|"GN"|None}
    - PN/GN onlyのformにはタグを付与
    - 複数typeに該当するformはambiguous_strategyに従う:
      - "skip": タグ付加しない (None)
      - "majority": 最頻typeを採用（未実装）
    """
    lex = pd.read_csv(lexicon_path)
    print(f"Loaded lexicon: {len(lex)} rows")

    # form→types集合
    form_types = {}
    for _, row in lex.iterrows():
        form = str(row["form"]).strip()
        t = str(row["type"]).strip()
        if form not in form_types:
            form_types[form] = set()
        form_types[form].add(t)

    # 統計
    tag_set = set(tag_types)
    n_unambiguous = 0
    n_ambiguous = 0
    n_non_target = 0

    form_tag_dict = {}
    for form, types in form_types.items():
        target_types = types & tag_set
        non_target_types = types - tag_set

        if len(target_types) == 1 and len(non_target_types) == 0:
            # 明確にPN or GNのみ
            form_tag_dict[form] = list(target_types)[0]
            n_unambiguous += 1
        elif len(target_types) >= 1 and len(non_target_types) >= 1:
            # PN/GN + word の混合 → ambiguous
            if ambiguous_strategy == "skip":
                form_tag_dict[form] = None
            n_ambiguous += 1
        else:
            # wordのみ or タグ対象外
            n_non_target += 1

    # Noneのエントリは除外（辞書に無い = タグ付加しない）
    form_tag_dict = {k: v for k, v in form_tag_dict.items() if v is not None}

    print(f"Unique forms: {len(form_types)}")
    print(f"Unambiguous PN/GN: {n_unambiguous}")
    print(f"Ambiguous (skipped): {n_ambiguous}")
    print(f"Non-target (word only): {n_non_target}")
    print(f"Final dict size: {len(form_tag_dict)}")

    # タグ別件数
    for tag in tag_types:
        count = sum(1 for v in form_tag_dict.values() if v == tag)
        print(f"  {tag}: {count} forms")

    return form_tag_dict


def main():
    os.chdir(EXP_DIR)
    tag_config = config["tagging"]
    lexicon_path = tag_config["lexicon_path"]
    tag_types = tag_config["tag_types"]
    ambiguous_strategy = tag_config["ambiguous_strategy"]

    form_tag_dict = build_form_type_dict(lexicon_path, tag_types, ambiguous_strategy)

    output_path = os.path.join(EXP_DIR, "dataset", "form_type_dict.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(form_tag_dict, f, ensure_ascii=False, indent=2)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
