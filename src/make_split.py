import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    p = argparse.ArgumentParser(description="Stratified 85/15 split for production holdout.")
    p.add_argument("--input", default="data/raw/loan_default.csv", help="Ruta al CSV original")
    p.add_argument("--target", default="Default", help="Columna target (0=bueno, 1=malo)")
    p.add_argument("--id-col", default="LoanID", help="Columna de ID (LoanID)")
    p.add_argument("--test-size", type=float, default=0.15, help="Proporción holdout (default 0.15)")
    p.add_argument("--out-train", default="data/processed/train.csv")
    p.add_argument("--out-test", default="data/processed/test_holdout.csv")
    p.add_argument("--out-metrics", default="data/processed/split_report.json")
    args = p.parse_args()

    in_path = Path(args.input)
    out_train = Path(args.out_train)
    out_test = Path(args.out_test)
    out_metrics = Path(args.out_metrics)

    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    if args.target not in df.columns:
        raise ValueError(f"No encuentro la columna target '{args.target}' en {in_path}.")
    if args.id_col not in df.columns:
        raise ValueError(f"No encuentro la columna ID '{args.id_col}' en {in_path}.")

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        stratify=df[args.target],
        random_state=42,
        shuffle=True,
    )

    train_df.to_csv(out_train, index=False)
    test_df.to_csv(out_test, index=False)

    def dist(y):
        vc = y.value_counts(dropna=False).sort_index()
        pct = (vc / len(y)).round(4).to_dict()
        return {"counts": vc.to_dict(), "proportions": pct}

    report = {
        "input_path": str(in_path),
        "rows_total": int(len(df)),
        "test_size": args.test_size,
        "target": args.target,
        "id_col": args.id_col,
        "train": {
            "rows": int(len(train_df)),
            "target_dist": dist(train_df[args.target]),
        },
        "test_holdout": {
            "rows": int(len(test_df)),
            "target_dist": dist(test_df[args.target]),
        },
    }

    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"✅ Train guardado en: {out_train}")
    print(f"✅ Holdout guardado en: {out_test}")
    print(f"📝 Reporte: {out_metrics}")

if __name__ == "__main__":
    main()