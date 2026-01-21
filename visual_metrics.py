import os
import pandas as pd
import matplotlib.pyplot as plt

METRICS = ["mae", "mse", "rmse", "mape", "mspe"]

# -----------------------------
# 0) Your raw data (as-is)
# -----------------------------
vlm_full = [
    ("ETTh1", "Ctx_512_Pred_96_dmodel_32", 0.392744, 0.363621, 0.60301, 9.032673, 35708.97),
    ("ETTh1", "Ctx_512_Pred_192_dmodel_32", 0.414021, 0.397256, 0.630283, 9.016186, 33273.02),
    ("ETTh1", "Ctx_512_Pred_336_dmodel_64", 0.435595, 0.420177, 0.648211, 10.019711, 38015.91),
    ("ETTh1", "Ctx_512_Pred_720_dmodel_256", 0.493356, 0.472826, 0.687624, 9.600866, 34137.516),

    ("Electricity", "Ctx_512_Pred_96_dmodel_128", 0.241177, 0.139134, 0.373007, 2.103261, 302393.84),
    ("Electricity", "Ctx_512_Pred_192_dmodel_128", 0.254532, 0.154231, 0.392722, 2.253603, 343315.6),
    ("Electricity", "Ctx_512_Pred_336_dmodel_256", 0.270536, 0.17062, 0.413061, 2.220086, 200647.08),
    ("Electricity", "Ctx_512_Pred_720_dmodel_64", 0.30171, 0.209057, 0.457228, 2.416697, 132297.47),

    ("Traffic", "Ctx_256_Pred_96_dmodel_128", 0.286968, 0.412577, 0.642321, 3.083977, 211767.08),
    ("Traffic", "Ctx_256_Pred_192_dmodel_128", 0.292469, 0.424896, 0.65184, 3.041433, 165870.89),
    ("Traffic", "Ctx_256_Pred_336_dmodel_256", 0.295625, 0.434511, 0.659175, 2.89705, 118293.555),
    ("Traffic", "Ctx_256_Pred_720_dmodel_512", 0.323022, 0.47476, 0.689028, 3.161536, 173918.7),

    ("Weather", "Ctx_256_Pred_96_dmodel_64", 0.202309, 0.155543, 0.394389, 12.98871, 20827240.0),
    ("Weather", "Ctx_256_Pred_192_dmodel_64", 0.242531, 0.193722, 0.440138, 13.778438, 20133500.0),
    ("Weather", "Ctx_256_Pred_336_dmodel_128", 0.285697, 0.246352, 0.496338, 14.139885, 22060464.0),
    ("Weather", "Ctx_256_Pred_720_dmodel_64", 0.331911, 0.315653, 0.56183, 14.637998, 23457088.0),
]

chronos2_full = [
    ("ETTh1", "Ctx_512_Pred_96_dmodel_32", 0.372648, 0.36209, 0.601739, 9.150147, 38124.95),
    ("ETTh1", "Ctx_512_Pred_192_dmodel_32", 0.407824, 0.414741, 0.644004, 10.244539, 43059.297),

    ("Electricity", "Ctx_512_Pred_96_dmodel_128", 0.206936, 0.123531, 0.351469, 1.86335, 128361.87),
    ("Electricity", "Ctx_512_Pred_192_dmodel_128", 0.226246, 0.143226, 0.378452, 1.974792, 94316.54),

    ("Traffic", "Ctx_256_Pred_96_dmodel_128", 0.225512, 0.371244, 0.609298, 2.046675, 49392.39),
    ("Traffic", "Ctx_256_Pred_192_dmodel_128", 0.233933, 0.392684, 0.626645, 2.121184, 47594.688),
    ("Traffic", "Ctx_256_Pred_336_dmodel_256", 0.241479, 0.408167, 0.63888, 2.113578, 41062.406),

    ("Weather", "Ctx_256_Pred_96_dmodel_64", 0.169919, 0.144758, 0.38047, 8.253313, 8333346.5),
    ("Weather", "Ctx_256_Pred_192_dmodel_64", 0.217718, 0.187607, 0.433137, 9.401886, 9169100.0),
    ("Weather", "Ctx_256_Pred_336_dmodel_128", 0.265629, 0.247406, 0.4974, 10.914833, 13236915.0),
    ("Weather", "Ctx_256_Pred_720_dmodel_64", 0.322131, 0.330159, 0.574595, 16.420654, 34915848.0),
]

vlm_zero = [
    ("ETTh1", "Ctx_256_Pred_96_dmodel_32", 0.87536, 1.608997, 1.268462, 20.80261, 147821.73),
    ("ETTh1", "Ctx_256_Pred_192_dmodel_32", 0.840695, 1.464499, 1.210165, 20.877903, 146763.55),
    ("ETTh1", "Ctx_256_Pred_336_dmodel_32", 0.655099, 0.916408, 0.957292, 15.397252, 78685.68),
    ("ETTh1", "Ctx_256_Pred_720_dmodel_32", 0.689635, 0.975678, 0.987764, 17.102507, 101205.516),

    ("Electricity", "Ctx_256_Pred_96_dmodel_32", 0.841196, 1.131401, 1.063673, 5.81258, 4008878.8),
    ("Electricity", "Ctx_256_Pred_192_dmodel_32", 0.846103, 1.132386, 1.064136, 5.616397, 3414294.5),
    ("Electricity", "Ctx_256_Pred_336_dmodel_32", 0.852383, 1.152236, 1.073423, 5.522578, 2733121.2),
    ("Electricity", "Ctx_256_Pred_720_dmodel_32", 0.873819, 1.21686, 1.103114, 5.655353, 2875508.2),

    ("Traffic", "Ctx_256_Pred_96_dmodel_128", 0.967497, 1.901103, 1.378805, 8.361523, 1609336.6),
    ("Traffic", "Ctx_256_Pred_192_dmodel_128", 0.965201, 1.897784, 1.377601, 7.992456, 1504569.1),
    ("Traffic", "Ctx_256_Pred_336_dmodel_256", 0.974145, 1.932955, 1.390307, 8.210604, 1537160.9),
    ("Traffic", "Ctx_256_Pred_720_dmodel_512", 0.972325, 1.959345, 1.399766, 7.829867, 1384915.9),

    ("Weather", "Ctx_256_Pred_96_dmodel_64", 0.329239, 0.292365, 0.540708, 17.41074, 28627448.0),
    ("Weather", "Ctx_256_Pred_192_dmodel_64", 0.352319, 0.328884, 0.573484, 17.684372, 31839000.0),
    ("Weather", "Ctx_256_Pred_336_dmodel_128", 0.383996, 0.381016, 0.617265, 18.084227, 34252312.0),
    ("Weather", "Ctx_256_Pred_720_dmodel_64", 0.414242, 0.431371, 0.656789, 18.916086, 38804544.0),
]

chronos2_zero = [
    ("ETTh1", "Ctx_256_Pred_96_dmodel_32", 0.386194, 0.399874, 0.632356, 9.982639, 47789.473),
    ("ETTh1", "Ctx_256_Pred_192_dmodel_32", 0.41564, 0.446879, 0.66849, 10.419199, 49261.227),
    ("ETTh1", "Ctx_256_Pred_336_dmodel_32", 0.434384, 0.478741, 0.691911, 10.410223, 49014.3),
    ("ETTh1", "Ctx_256_Pred_720_dmodel_32", 0.450998, 0.467526, 0.683759, 10.961671, 55886.344),

    
    ("Electricity", "Ctx_256_Pred_96_dmodel_32", 0.220249, 0.136387, 0.369306, 2.169856, 258713.36),
    ("Electricity", "Ctx_256_Pred_192_dmodel_32", 0.235214, 0.152363, 0.390337, 2.280325, 281879.38),
    ("Electricity", "Ctx_256_Pred_336_dmodel_32", 0.254738, 0.174727, 0.418003, 2.326266, 217335.69),
    ("Electricity", "Ctx_256_Pred_720_dmodel_32", 0.292004, 0.259277, 0.509192, 2.541622, 150370.6),
]

# -----------------------------
# 1) Build DataFrames (simple)
# -----------------------------
METRICS = ["mae", "mse", "rmse", "mape", "mspe"]

def make_df(rows, model_name: str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["dataset", "case", "mae", "mse", "rmse", "mape", "mspe"])
    df["model"] = model_name
    return df

df_vlm_full = make_df(vlm_full, "VLM")
df_chr_full = make_df(chronos2_full, "Chronos2")
df_vlm_zero = make_df(vlm_zero, "VLM")
df_chr_zero = make_df(chronos2_zero, "Chronos2")


# -----------------------------
# 2) Pair by common (dataset, case)
#    - full-shot: only keep first 2 cases per dataset (after pairing)
#    - zero-shot: keep all common cases
# -----------------------------
def common_keys(df_a: pd.DataFrame, df_b: pd.DataFrame):
    a = set(zip(df_a["dataset"], df_a["case"]))
    b = set(zip(df_b["dataset"], df_b["case"]))
    return a.intersection(b)

def filter_by_keys(df: pd.DataFrame, keys_set: set) -> pd.DataFrame:
    return df[df.apply(lambda r: (r["dataset"], r["case"]) in keys_set, axis=1)].copy()

# Full-shot common
keys_full = common_keys(df_vlm_full, df_chr_full)
df_full_common = pd.concat(
    [filter_by_keys(df_vlm_full, keys_full), filter_by_keys(df_chr_full, keys_full)],
    ignore_index=True
).sort_values(["dataset", "case", "model"]).reset_index(drop=True)

# print all common keys
for k in sorted(keys_full):
    print(f"[FULL-SHOT COMMON KEY] dataset={k[0]}, case={k[1]}")

# head 10 of full common before filtering
print(df_full_common.head(20))

# Keep first 2 cases per dataset (based on lexicographic order of "case")
keep_map = {}
for ds in df_full_common["dataset"].unique():
    cases = sorted(df_full_common[df_full_common["dataset"] == ds]["case"].unique().tolist())
    keep_map[ds] = set(cases[:2])

df_full_common = df_full_common[
    df_full_common.apply(lambda r: r["case"] in keep_map[r["dataset"]], axis=1)
].copy().reset_index(drop=True)

# head 10 of full common before filtering
print("After filtering to first 2 cases per dataset:")
print(df_full_common.head(20))

# Zero-shot common
keys_zero = common_keys(df_vlm_zero, df_chr_zero)
df_zero_common = pd.concat(
    [filter_by_keys(df_vlm_zero, keys_zero), filter_by_keys(df_chr_zero, keys_zero)],
    ignore_index=True
).sort_values(["dataset", "case", "model"]).reset_index(drop=True)

# print all common keys
for k in sorted(keys_zero):
    print(f"[ZERO-SHOT COMMON KEY] dataset={k[0]}, case={k[1]}")

# -----------------------------
# 3) Plot by METRIC (one figure per metric)
#    Each x-tick = dataset_case, each tick has 2 bars (VLM vs Chronos2)
#    + annotate values on bars
# -----------------------------
def _format_value(v: float) -> str:
    # pretty formatting: keep readability across magnitudes
    if pd.isna(v):
        return ""
    av = abs(float(v))
    if av >= 10000:
        return f"{v:.0f}"
    if av >= 100:
        return f"{v:.1f}"
    if av >= 1:
        return f"{v:.3f}"
    return f"{v:.4f}"

def _annotate_bars(ax, bars, values, y_offset_ratio=0.01):
    """
    在每个 bar 顶部标注数值（横向显示）
    """
    ylim = ax.get_ylim()
    y_range = (ylim[1] - ylim[0]) if (ylim[1] - ylim[0]) != 0 else 1.0
    offset = y_offset_ratio * y_range

    for rect, val in zip(bars, values):
        if pd.isna(val):
            continue

        h = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            h + offset,
            _format_value(val),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,        # ✅ 横着
            clip_on=True,
        )

def plot_metric_compare(df_two_models: pd.DataFrame, metric: str,
                        title_prefix: str, outdir: str, show: bool = True):
    os.makedirs(outdir, exist_ok=True)

    df = df_two_models.copy()
    df["label"] = df["dataset"].astype(str) + "_" + df["case"].astype(str)

    piv = df.pivot_table(index="label", columns="model", values=metric, aggfunc="mean")

    for col in ["VLM", "Chronos2"]:
        if col not in piv.columns:
            print(f"[SKIP] metric={metric}: missing column {col} in pivot.")
            return
    piv = piv.dropna(subset=["VLM", "Chronos2"])
    if piv.empty:
        print(f"[SKIP] metric={metric}: no paired rows after dropna.")
        return

    piv = piv.sort_index()

    labels = piv.index.tolist()
    x = list(range(len(labels)))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(12, 0.45 * len(labels)), 6))

    bars_vlm = ax.bar([i - width/2 for i in x], piv["VLM"].values, width=width, label="VLM")
    bars_chr = ax.bar([i + width/2 for i in x], piv["Chronos2"].values, width=width, label="Chronos2")

    ax.set_ylabel(metric)
    ax.set_title(f"{title_prefix} — {metric}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.legend(frameon=False)

    # IMPORTANT: set ylim first, then annotate (so offsets are stable)
    ax.relim()
    ax.autoscale_view()

    _annotate_bars(ax, bars_vlm, piv["VLM"].values)
    _annotate_bars(ax, bars_chr, piv["Chronos2"].values)

    fig.tight_layout()

    safe_metric = metric.replace("/", "_")
    path = os.path.join(outdir, f"{title_prefix.replace(' ', '_')}__{safe_metric}.png")
    fig.savefig(path, dpi=200)
    print(f"[OK] Saved: {path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_all_metrics(df_two_models: pd.DataFrame, title_prefix: str, outdir: str, show: bool = True):
    for m in METRICS:
        plot_metric_compare(df_two_models, m, title_prefix=title_prefix, outdir=outdir, show=show)


# -----------------------------
# 4) Run
# -----------------------------
if __name__ == "__main__":
    plot_all_metrics(df_full_common, title_prefix="FULL-SHOT VLM vs Chronos2", outdir="plots_fullshot_by_metric", show=True)
    plot_all_metrics(df_zero_common, title_prefix="ZERO-SHOT VLM vs Chronos2", outdir="plots_zeroshot_by_metric", show=True)
