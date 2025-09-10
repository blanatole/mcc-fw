#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support


def load_preds(results_dir, prefix):
    # Expect preds file from run_mm_late with prob_pos column in saved predictions
    # If not present, we will instruct user how to regenerate.
    paths = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.startswith(prefix) and f.endswith("preds.csv")
    ]
    if not paths:
        raise FileNotFoundError("Không tìm thấy file preds.csv phù hợp trong results.")
    # Pick latest by mtime
    path = max(paths, key=os.path.getmtime)
    df = pd.read_csv(path)
    return df, path


def tune_threshold(y_true, prob_pos, start=0.30, end=0.70, step=0.02):
    best_thr, best_f1 = None, -1.0
    rows = []
    thr = start
    while thr <= end + 1e-8:
        y_pred = (prob_pos >= thr).astype(int)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
        rows.append({
            "threshold": round(thr, 4),
            "f1_macro": f1_macro,
            "precision_macro": p_macro,
            "recall_macro": r_macro,
        })
        if f1_macro > best_f1:
            best_f1, best_thr = f1_macro, thr
        thr += step
    return best_thr, best_f1, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/mm_late/", help="Thư mục kết quả")
    parser.add_argument("--prefix", required=True, help="Tiền tố file, ví dụ: bertweet-vit-concat_task8_seed30_itc0.1_")
    parser.add_argument("--out", default="results/mm_late/threshold_search.csv")
    args = parser.parse_args()

    df, path = load_preds(args.results_dir, args.prefix)
    if "prob_pos" not in df.columns:
        raise ValueError(f"File {path} không có cột prob_pos. Hãy chạy lại với bản code đã cập nhật để lưu xác suất.")

    # y_true có thể ở dạng one-hot; đảm bảo về nhãn 0/1
    y_true = df["label"].values
    if isinstance(y_true[0], str) and y_true[0].startswith("["):
        # chuỗi list -> parse
        y_true = df["label"].apply(lambda s: int(np.argmax(eval(s)))).values

    prob_pos = df["prob_pos"].values.astype(float)

    best_thr, best_f1, df_search = tune_threshold(y_true, prob_pos)
    df_search.to_csv(args.out, index=False)

    print(f"Best threshold: {best_thr:.2f} | Best F1-macro: {best_f1:.4f}")
    print(f"Saved search to: {args.out}")


if __name__ == "__main__":
    main()


