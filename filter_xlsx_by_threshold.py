#!/usr/bin/env python3
"""指定列の値がしきい値を超える行だけを抽出してxlsx保存する。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="指定した列の値がしきい値を超える行のみを抽出して新しいxlsxを作成します。"
    )
    parser.add_argument("--input", required=True, help="入力xlsxファイルのパス")
    parser.add_argument("--output", required=True, help="出力xlsxファイルのパス")
    parser.add_argument("--column", required=True, help="判定に使う列名 (例: H2O)")
    parser.add_argument("--threshold", type=float, required=True, help="しきい値")
    parser.add_argument(
        "--operator",
        choices=[">", ">=", "<", "<="],
        default=">",
        help="比較演算子 (デフォルト: >)",
    )
    parser.add_argument(
        "--sheet",
        default=0,
        help="読み込むシート名またはインデックス (デフォルト: 0)",
    )
    parser.add_argument(
        "--include-equal",
        action="store_true",
        help="後方互換オプション。指定時、--operator 未指定なら >= として扱います。",
    )
    return parser.parse_args()


def _parse_sheet(sheet_value: str | int):
    if isinstance(sheet_value, int):
        return sheet_value
    if isinstance(sheet_value, str) and sheet_value.isdigit():
        return int(sheet_value)
    return sheet_value


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_path}")

    sheet = _parse_sheet(args.sheet)
    xls = pd.ExcelFile(input_path)
    all_sheets = {name: xls.parse(sheet_name=name) for name in xls.sheet_names}

    if isinstance(sheet, int):
        if sheet < 0 or sheet >= len(xls.sheet_names):
            raise ValueError(
                f"シートインデックス {sheet} は範囲外です。0 から {len(xls.sheet_names) - 1} を指定してください。"
            )
        target_sheet_name = xls.sheet_names[sheet]
    else:
        target_sheet_name = str(sheet)
        if target_sheet_name not in all_sheets:
            raise ValueError(
                f"シート '{target_sheet_name}' が見つかりません。利用可能: {', '.join(xls.sheet_names)}"
            )

    df = all_sheets[target_sheet_name]

    if args.column not in df.columns:
        available = ", ".join(map(str, df.columns))
        raise ValueError(
            f"列 '{args.column}' が存在しません。利用可能な列: {available}"
        )

    values = pd.to_numeric(df[args.column], errors="coerce")
    op = args.operator
    if args.include_equal and op == ">":
        op = ">="

    if op == ">":
        mask = values > args.threshold
    elif op == ">=":
        mask = values >= args.threshold
    elif op == "<":
        mask = values < args.threshold
    elif op == "<=":
        mask = values <= args.threshold
    else:
        raise ValueError(f"未対応の演算子です: {op}")

    filtered = df[mask.fillna(False)].copy()
    all_sheets[target_sheet_name] = filtered

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name in xls.sheet_names:
            all_sheets[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"入力: {input_path}")
    print(f"出力: {output_path}")
    print(f"対象シート: {target_sheet_name}")
    print(f"条件: {args.column} {op} {args.threshold}")
    print(f"抽出件数: {len(filtered)} / {len(df)}")


if __name__ == "__main__":
    main()
