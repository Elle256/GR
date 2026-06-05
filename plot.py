"""
plot.py – Vẽ tất cả biểu đồ từ kết quả thí nghiệm.

Biểu đồ được tạo:
  1. Accuracy theo rounds       → images/convergence_<alpha>_<dropout>.png
  2. Heatmap α × dropout        → images/heatmap_accuracy.png
  3. Bar chart FedAvg vs FedProx → images/compare_bar.png
  4. Client drift theo rounds   → images/drift_<alpha>_<dropout>.png
  (Bonus) Tất cả trong 1 figure  → images/summary_grid.png

Cách dùng:
  python plot.py                        # đọc experiments/results/*.json
  python plot.py --results_dir path/to  # chỉ định thư mục khác
  python plot.py --alpha 0.1 --dropout 0.4  # chỉ vẽ một điều kiện cụ thể
"""

import os
import json
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")   # không cần màn hình (headless)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Style toàn cục
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "fedavg":  "#378ADD",   # xanh dương
    "fedprox": "#D85A30",   # cam san hô
}
ALPHA_MARKERS = {0.1: "o", 0.5: "s", 1.0: "^", 10.0: "D"}
DROPOUT_LINESTYLES = {0.0: "-", 0.2: "--", 0.4: "-.", 0.6: ":"}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "lines.linewidth":  2,
    "figure.dpi":       150,
    "savefig.bbox":     "tight",
    "savefig.facecolor":"white",
})


# ─────────────────────────────────────────────────────────────────────────────
# Đọc và tổ chức dữ liệu
# ─────────────────────────────────────────────────────────────────────────────

def load_results(results_dir: str) -> List[dict]:
    """Đọc tất cả file JSON trong results_dir."""
    pattern = os.path.join(results_dir, "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[plot.py] Không tìm thấy file nào trong '{results_dir}'")
        return []
    results = []
    for f in files:
        try:
            with open(f) as fp:
                results.append(json.load(fp))
        except Exception as e:
            print(f"[plot.py] Lỗi đọc {f}: {e}")
    print(f"[plot.py] Đọc được {len(results)} file kết quả.")
    return results


def group_results(results: List[dict]) -> Dict:
    """
    Nhóm kết quả theo (algorithm, alpha, dropout).
    Trả về dict: {(algo, alpha, dropout): [list of run dicts]}
    """
    grouped = defaultdict(list)
    for r in results:
        cfg = r["config"]
        key = (cfg["algorithm"], float(cfg["alpha"]), float(cfg["dropout"]))
        grouped[key].append(r)
    return grouped


def mean_history(runs: List[dict], metric: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tính trung bình + std của một metric qua nhiều seed.
    Trả về (rounds, mean, std).
    """
    all_vals = [r["history"][metric] for r in runs if metric in r["history"]]
    # Căn cho cùng độ dài (lấy min)
    min_len = min(len(v) for v in all_vals)
    arr = np.array([v[:min_len] for v in all_vals])
    rounds = np.array(runs[0]["history"]["rounds"][:min_len])
    return rounds, arr.mean(axis=0), arr.std(axis=0)


def mean_summary(runs: List[dict], metric: str) -> Tuple[float, float]:
    """Trả về (mean, std) của một summary metric qua nhiều seed."""
    vals = [r["summary"][metric] for r in runs if metric in r.get("summary", {})]
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


# ─────────────────────────────────────────────────────────────────────────────
# Biểu đồ 1 – Accuracy theo rounds (đường hội tụ)
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(
    grouped: Dict,
    save_dir: str,
    filter_alpha: Optional[float] = None,
    filter_dropout: Optional[float] = None,
):
    """
    Vẽ test accuracy theo rounds cho FedAvg vs FedProx.
    Mỗi điều kiện (alpha, dropout) ra một file riêng.
    """
    # Lấy tất cả (alpha, dropout) cần vẽ
    conditions = set()
    for (algo, alpha, dropout) in grouped:
        conditions.add((alpha, dropout))

    if filter_alpha is not None:
        conditions = {c for c in conditions if c[0] == filter_alpha}
    if filter_dropout is not None:
        conditions = {c for c in conditions if c[1] == filter_dropout}

    os.makedirs(save_dir, exist_ok=True)

    for (alpha, dropout) in sorted(conditions):
        fig, ax = plt.subplots(figsize=(7, 4.5))

        for algo in ["fedavg", "fedprox"]:
            key = (algo, alpha, dropout)
            if key not in grouped:
                continue
            rounds, mean_acc, std_acc = mean_history(grouped[key], "test_acc")
            mean_pct = mean_acc * 100
            std_pct  = std_acc  * 100

            label = "FedAvg" if algo == "fedavg" else f"FedProx (μ={grouped[key][0]['config']['mu']})"
            ax.plot(rounds, mean_pct, color=PALETTE[algo], label=label)
            ax.fill_between(rounds,
                            mean_pct - std_pct,
                            mean_pct + std_pct,
                            alpha=0.15, color=PALETTE[algo])

        ax.set_title(f"Hội tụ accuracy  |  α={alpha},  dropout={int(dropout*100)}%")
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Test accuracy (%)")
        ax.legend(framealpha=0.8)
        ax.set_ylim(bottom=0)

        fname = f"convergence_alpha{alpha}_drop{int(dropout*100)}.png"
        fig.savefig(os.path.join(save_dir, fname))
        plt.close(fig)
        print(f"  [1] Lưu: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Biểu đồ 2 – Heatmap accuracy theo α × dropout
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(grouped: Dict, save_dir: str):
    """
    2 heatmap cạnh nhau: FedAvg (trái) và FedProx (phải).
    Trục X = dropout rate, Trục Y = alpha.
    Màu = best test accuracy (%).
    """
    alphas   = sorted({k[1] for k in grouped})
    dropouts = sorted({k[2] for k in grouped})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    vmin, vmax = 100.0, 0.0

    # Thu thập dữ liệu cho cả hai để đồng nhất colorscale
    data_all = {}
    for algo in ["fedavg", "fedprox"]:
        mat = np.full((len(alphas), len(dropouts)), np.nan)
        for i, alpha in enumerate(alphas):
            for j, drop in enumerate(dropouts):
                key = (algo, alpha, drop)
                if key in grouped:
                    m, _ = mean_summary(grouped[key], "best_acc")
                    mat[i, j] = m * 100
                    vmin = min(vmin, mat[i, j])
                    vmax = max(vmax, mat[i, j])
        data_all[algo] = mat

    vmin = max(0, vmin - 2)   # padding dưới
    vmax = min(100, vmax + 1) # padding trên

    for ax, algo in zip(axes, ["fedavg", "fedprox"]):
        mat = data_all[algo]
        im = ax.imshow(mat, vmin=vmin, vmax=vmax,
                       cmap="RdYlGn", aspect="auto", origin="lower")

        ax.set_xticks(range(len(dropouts)))
        ax.set_xticklabels([f"{int(d*100)}%" for d in dropouts])
        ax.set_yticks(range(len(alphas)))
        ax.set_yticklabels([str(a) for a in alphas])
        ax.set_xlabel("Dropout rate")
        ax.set_ylabel("Dirichlet α  (↓ = more Non-IID)")
        ax.set_title("FedAvg" if algo == "fedavg" else "FedProx")
        ax.grid(False)

        # Ghi số vào từng ô
        for i in range(len(alphas)):
            for j in range(len(dropouts)):
                val = mat[i, j]
                if not np.isnan(val):
                    txt_color = "white" if val < (vmin + vmax) / 2 else "black"
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            fontsize=10, color=txt_color, fontweight="bold")

    fig.colorbar(im, ax=axes, label="Best test accuracy (%)", shrink=0.85)
    fig.suptitle("Best accuracy  ×  (α, dropout)", y=1.02, fontsize=13)

    fname = "heatmap_accuracy.png"
    fig.savefig(os.path.join(save_dir, fname))
    plt.close(fig)
    print(f"  [2] Lưu: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Biểu đồ 3 – Bar chart FedAvg vs FedProx
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_bar(grouped: Dict, save_dir: str):
    """
    Bar chart so sánh best accuracy của FedAvg vs FedProx
    cho từng điều kiện (alpha, dropout).
    Các thanh được nhóm theo alpha, mỗi nhóm có các dropout khác nhau.
    """
    alphas   = sorted({k[1] for k in grouped})
    dropouts = sorted({k[2] for k in grouped})

    # Mỗi subplot = 1 alpha value
    n_cols = 2
    n_rows = (len(alphas) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(13, 4.5 * n_rows),
                             sharey=False)
    axes = np.array(axes).flatten()

    bar_width = 0.35
    x = np.arange(len(dropouts))

    for idx, alpha in enumerate(alphas):
        ax = axes[idx]

        for algo_idx, algo in enumerate(["fedavg", "fedprox"]):
            means, stds = [], []
            for drop in dropouts:
                key = (algo, alpha, drop)
                if key in grouped:
                    m, s = mean_summary(grouped[key], "best_acc")
                    means.append(m * 100)
                    stds.append(s * 100)
                else:
                    means.append(0.0)
                    stds.append(0.0)

            offset = (algo_idx - 0.5) * bar_width
            bars = ax.bar(x + offset, means, bar_width,
                          yerr=stds, capsize=4,
                          color=PALETTE[algo], alpha=0.85,
                          label="FedAvg" if algo == "fedavg" else "FedProx",
                          error_kw={"elinewidth": 1.2})

            # Ghi số trên cột
            for bar, m in zip(bars, means):
                if m > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.5,
                            f"{m:.1f}", ha="center", va="bottom",
                            fontsize=9, color="#333")

        ax.set_title(f"α = {alpha}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"drop {int(d*100)}%" for d in dropouts])
        ax.set_ylabel("Best accuracy (%)")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9)

    # Ẩn subplot thừa nếu có
    for idx in range(len(alphas), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("FedAvg vs FedProx – Best accuracy theo điều kiện", fontsize=14, y=1.01)
    fig.tight_layout()

    fname = "compare_bar.png"
    fig.savefig(os.path.join(save_dir, fname))
    plt.close(fig)
    print(f"  [3] Lưu: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Biểu đồ 4 – Client drift theo rounds
# ─────────────────────────────────────────────────────────────────────────────

def plot_drift(
    grouped: Dict,
    save_dir: str,
    filter_alpha: Optional[float] = None,
    filter_dropout: Optional[float] = None,
):
    conditions = set()
    for (algo, alpha, dropout) in grouped:
        conditions.add((alpha, dropout))

    if filter_alpha is not None:
        conditions = {c for c in conditions if c[0] == filter_alpha}
    if filter_dropout is not None:
        conditions = {c for c in conditions if c[1] == filter_dropout}

    conditions = sorted(conditions)
    if not conditions:
        print("  [!] Không có điều kiện nào phù hợp.")
        return

    os.makedirs(save_dir, exist_ok=True)

    # --- Layout grid: hàng = alpha, cột = dropout ---
    alphas   = sorted(set(c[0] for c in conditions))
    dropouts = sorted(set(c[1] for c in conditions))
    n_rows, n_cols = len(alphas), len(dropouts)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        sharex=False, sharey=False,
        squeeze=False,
    )

    fig.suptitle(
        "Client Drift theo rounds  —  FedAvg vs FedProx",
        fontsize=14, fontweight="bold", y=1.01
    )

    for row, alpha in enumerate(alphas):
        for col, dropout in enumerate(dropouts):
            ax = axes[row][col]

            if (alpha, dropout) not in conditions:
                ax.set_visible(False)
                continue

            has_data = False
            for algo in ["fedavg", "fedprox"]:
                key = (algo, alpha, dropout)
                if key not in grouped:
                    continue
                rounds, mean_d, std_d = mean_history(grouped[key], "client_drift")

                label = "FedAvg" if algo == "fedavg" else "FedProx"
                ax.plot(rounds, mean_d, color=PALETTE[algo], label=label, linewidth=1.8)
                ax.fill_between(
                    rounds,
                    mean_d - std_d,
                    mean_d + std_d,
                    alpha=0.15, color=PALETTE[algo]
                )
                has_data = True

            if not has_data:
                ax.text(0.5, 0.5, "Chưa có dữ liệu",
                        ha="center", va="center", transform=ax.transAxes,
                        color="gray", fontsize=10)

            ax.set_title(f"α={alpha},  dropout={int(dropout*100)}%", fontsize=10)
            ax.set_ylim(bottom=0)

            # Chỉ label trục ngoài cùng để gọn
            if row == n_rows - 1:
                ax.set_xlabel("Communication round")
            if col == 0:
                ax.set_ylabel("Γ = ||w_k − w_global||")

            ax.legend(fontsize=8, framealpha=0.8)

    fig.tight_layout()
    fname = "drift_comparison_grid.png"
    fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [4] Lưu: {fname}")

# ─────────────────────────────────────────────────────────────────────────────
# Bonus – Summary grid (4 điều kiện tiêu biểu trong 1 figure)
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_grid(grouped: Dict, save_dir: str):
    """
    1 figure lớn gồm 4 subplots (2×2):
    Chọn 4 điều kiện tiêu biểu: (α=0.1, drop=0%), (α=0.1, drop=40%),
                                  (α=10,  drop=0%), (α=10,  drop=40%)
    Mỗi subplot vẽ accuracy hội tụ của FedAvg vs FedProx.
    """
    conditions = [
        (0.1, 0.0, "Rất Non-IID, không dropout"),
        (0.1, 0.4, "Rất Non-IID, dropout 40%"),
        (10.0, 0.0, "Gần IID, không dropout"),
        (10.0, 0.4, "Gần IID, dropout 40%"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, (alpha, dropout, title) in zip(axes, conditions):
        has_data = False
        for algo in ["fedavg", "fedprox"]:
            key = (algo, alpha, dropout)
            if key not in grouped:
                continue
            rounds, mean_acc, std_acc = mean_history(grouped[key], "test_acc")
            mean_pct = mean_acc * 100
            std_pct  = std_acc  * 100
            label = "FedAvg" if algo == "fedavg" else "FedProx"
            ax.plot(rounds, mean_pct, color=PALETTE[algo], label=label)
            ax.fill_between(rounds, mean_pct - std_pct, mean_pct + std_pct,
                            alpha=0.15, color=PALETTE[algo])
            has_data = True

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Round")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_ylim(0, 100)
        if has_data:
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, "Chưa có dữ liệu", transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontsize=12)

    fig.suptitle("Tóm tắt: FedAvg vs FedProx – 4 điều kiện tiêu biểu",
                 fontsize=14, y=1.01)
    fig.tight_layout()

    fname = "summary_grid.png"
    fig.savefig(os.path.join(save_dir, fname))
    plt.close(fig)
    print(f"  [+] Lưu: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ từ kết quả FL")
    parser.add_argument("--results_dir", default="experiments/results",
                        help="Thư mục chứa file JSON kết quả")
    parser.add_argument("--save_dir", default="images",
                        help="Thư mục lưu ảnh PNG")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Chỉ vẽ một alpha cụ thể (vd: 0.1)")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Chỉ vẽ một dropout rate cụ thể (vd: 0.4)")
    parser.add_argument("--charts", nargs="+",
                        default=["convergence", "heatmap", "bar", "drift", "summary"],
                        choices=["convergence", "heatmap", "bar", "drift", "summary"],
                        help="Chọn biểu đồ cần vẽ")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  plot.py – Vẽ biểu đồ FL")
    print(f"  Đọc từ : {args.results_dir}")
    print(f"  Lưu vào: {args.save_dir}")
    print(f"{'='*55}\n")

    results = load_results(args.results_dir)
    if not results:
        print("Không có dữ liệu. Hãy chạy launch.sh trước.")
        return

    grouped = group_results(results)
    print(f"  Tìm thấy {len(grouped)} tổ hợp (algo, α, dropout) khác nhau.\n")

    if "convergence" in args.charts:
        print("[1] Vẽ accuracy hội tụ...")
        plot_convergence(grouped, args.save_dir, args.alpha, args.dropout)

    if "heatmap" in args.charts:
        print("[2] Vẽ heatmap...")
        plot_heatmap(grouped, args.save_dir)

    if "bar" in args.charts:
        print("[3] Vẽ bar chart so sánh...")
        plot_comparison_bar(grouped, args.save_dir)

    if "drift" in args.charts:
        print("[4] Vẽ client drift...")
        plot_drift(grouped, args.save_dir, args.alpha, args.dropout)

    if "summary" in args.charts:
        print("[+] Vẽ summary grid...")
        plot_summary_grid(grouped, args.save_dir)

    print(f"\nXong! Tất cả biểu đồ đã lưu vào '{args.save_dir}/'")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()