#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_expected_to_ns(expected: str) -> int | None:
    """
    expected: like '4h', '15min', '1d', '240m', '3600s'
    Returns expected delta in nanoseconds, or None.
    """
    if expected is None:
        return None
    expected = expected.strip()
    if expected == "":
        return None
    try:
        td = pd.to_timedelta(expected)
        return int(td.value)  # ns
    except Exception:
        return None


def read_eventtime(csv_path: str, col: str, nrows: int | None, usecols_extra: list[str] | None):
    # 只读必要列，速度快；你也可以加上 symbol/interval 之类列用于定位
    usecols = [col]
    if usecols_extra:
        for c in usecols_extra:
            if c not in usecols:
                usecols.append(c)

    df = pd.read_csv(
        csv_path,
        usecols=usecols,
        nrows=nrows,
        low_memory=False,
    )
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    # eventtime 是 ms epoch（你给的样例是 1574928000000）
    # 若列里混有字符串/空值，这里会尽量转成 numeric 再 to_datetime
    et = pd.to_numeric(df[col], errors="coerce")
    bad = et.isna().sum()
    if bad > 0:
        print(f"[WARN] {bad} rows have non-numeric/NaN {col}. They will be dropped for checks.")
    df = df.loc[~et.isna()].copy()
    df[col] = et.loc[~et.isna()].astype("int64")

    # 转成 UTC datetime（tz-aware），同时给一个 naive UTC 版本便于展示/比较
    df["__dt_utc"] = pd.to_datetime(df[col], unit="ms", utc=True)
    df["__dt"] = df["__dt_utc"].dt.tz_convert(None)  # naive-UTC

    return df


def analyze_sequence(df: pd.DataFrame, label: str, expected_ns: int | None, topk: int):
    """
    Analyze diffs in the current row order of df (no sorting here).
    """
    idx = pd.DatetimeIndex(df["__dt"])
    n = len(idx)

    print(f"\n=== [{label}] ===")
    print(f"rows: {n}")
    print(f"monotonic_increasing: {idx.is_monotonic_increasing}")
    # duplicates by timestamp
    dup_mask = pd.Series(idx).duplicated(keep=False).to_numpy()
    n_dup = int(dup_mask.sum())
    print(f"duplicate timestamps (count, incl. all occurrences): {n_dup}")

    if n < 2:
        print("Not enough rows to compute diffs.")
        return None

    # diffs in ns
    diffs = np.diff(idx.asi8)  # ns
    # basic stats
    print(f"diff stats (ns): min={diffs.min()}  max={diffs.max()}  unique={len(np.unique(diffs))}")

    # top-k most common diffs
    vals, counts = np.unique(diffs, return_counts=True)
    order = np.argsort(-counts)
    print(f"\nTop {min(topk, len(vals))} most common diffs:")
    for i in order[:topk]:
        td = pd.to_timedelta(int(vals[i]), unit="ns")
        print(f"  {str(td):>16}  count={int(counts[i])}")

    # identify anomalies
    anomalies = np.zeros_like(diffs, dtype=bool)

    if expected_ns is not None:
        anomalies = diffs != expected_ns
        exp_td = pd.to_timedelta(expected_ns, unit="ns")
        print(f"\nExpected diff: {exp_td}. Anomalies (diff != expected): {int(anomalies.sum())} / {len(diffs)}")
    else:
        # 没指定 expected：把“非众数”视为异常
        mode_val = vals[order[0]]
        mode_td = pd.to_timedelta(int(mode_val), unit="ns")
        anomalies = diffs != mode_val
        print(f"\nMode diff: {mode_td}. Non-mode anomalies: {int(anomalies.sum())} / {len(diffs)}")

    # show a few anomaly examples (with prev/next timestamps)
    if anomalies.any():
        # anomalies at position i means between row i and i+1
        pos = np.where(anomalies)[0]
        show = pos[:20]  # show first 20
        print("\nFirst anomaly examples (showing up to 20):")
        for i in show:
            t0 = idx[i]
            t1 = idx[i + 1]
            td = pd.to_timedelta(int(diffs[i]), unit="ns")
            print(f"  at rows {i} -> {i+1}: {t0}  ->  {t1}   diff={td}")
    else:
        print("\nNo anomalies found under this criterion.")

    return diffs


def save_anomaly_report(df_sorted: pd.DataFrame, expected_ns: int | None, out_prefix: str):
    """
    Save a CSV with anomaly boundaries in sorted order: rows i and i+1 where diff is anomalous.
    """
    idx = pd.DatetimeIndex(df_sorted["__dt"])
    if len(idx) < 2:
        return

    diffs = np.diff(idx.asi8)
    if expected_ns is not None:
        mask = diffs != expected_ns
        crit = f"diff != {pd.to_timedelta(expected_ns, unit='ns')}"
    else:
        vals, counts = np.unique(diffs, return_counts=True)
        mode_val = vals[np.argmax(counts)]
        mask = diffs != mode_val
        crit = f"diff != mode({pd.to_timedelta(int(mode_val), unit='ns')})"

    pos = np.where(mask)[0]
    if len(pos) == 0:
        return

    rows = []
    for i in pos:
        rows.append({
            "i": int(i),
            "eventtime_i_ms": int(df_sorted.iloc[i]["eventtime"]),
            "eventtime_ip1_ms": int(df_sorted.iloc[i+1]["eventtime"]),
            "dt_i": str(idx[i]),
            "dt_ip1": str(idx[i+1]),
            "diff": str(pd.to_timedelta(int(diffs[i]), unit="ns")),
        })

    out_csv = f"{out_prefix}_anomalies_sorted.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n[Saved] {out_csv}  (criterion: {crit}, rows={len(rows)})")


def main():
    ap = argparse.ArgumentParser(description="Check irregular intervals in CSV eventtime (ms epoch).")
    ap.add_argument("--csv", required=True, help="Path to CSV file")
    ap.add_argument("--col", default="eventtime", help="Timestamp column name (default: eventtime)")
    ap.add_argument("--expected", default=None, help="Expected interval like '4h', '15min'. If omitted, use mode diff as reference.")
    ap.add_argument("--topk", type=int, default=10, help="Top-K diffs to show")
    ap.add_argument("--nrows", type=int, default=None, help="Read only first N rows (debug)")
    ap.add_argument("--out", default=None, help="Output prefix for anomaly report CSV (sorted order)")
    ap.add_argument("--extra", nargs="*", default=["symbol", "interval"], help="Extra columns to load if exist (for context)")
    args = ap.parse_args()

    csv_path = args.csv
    expected_ns = parse_expected_to_ns(args.expected)

    # Load
    extra = args.extra or []
    # 如果 extra 列不存在，read_csv 会报错；这里先尝试只读 eventtime，再尝试补充（更稳）
    try:
        df = read_eventtime(csv_path, args.col, args.nrows, extra)
    except ValueError:
        df = read_eventtime(csv_path, args.col, args.nrows, usecols_extra=None)

    # For report function, it expects column name eventtime; normalize:
    if args.col != "eventtime":
        df = df.rename(columns={args.col: "eventtime"})

    # Analyze in original order (as-is in CSV)
    analyze_sequence(df, "as-is CSV order", expected_ns, args.topk)

    # Analyze after sorting by time (helps distinguish out-of-order vs true gaps)
    df_sorted = df.sort_values("__dt").reset_index(drop=True)
    analyze_sequence(df_sorted, "sorted by eventtime", expected_ns, args.topk)

    # Save anomaly boundary pairs in sorted order
    if args.out:
        save_anomaly_report(df_sorted, expected_ns, args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()
