import math
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from copy import deepcopy

import argparse


class TrainVisualizer:
    auc_candidates = [
        "val_auc", "train_auc", "auc",
    ]

    sub_candidates = [
        "val_pr_auc", "pr_auc",
        "val_sensitivity", "sensitivity",
        "val_specificity", "specificity",
        "val_f1", "f1",
    ]

    # default
    experiment_root = "./results"
    run_log_name = "run_logging.json"      

    # phase color
    PHASE_KEY = "extreme"
    PHASE_COLORS = {
        "pastel": [
            "#F3F4F6",  # gray-100
            "#ECFEFF",  # cyan-50
            "#F0FDF4",  # green-50
            "#FFF7ED",  # orange-50
            "#EEF2FF",  # indigo-50
            "#FAF5FF",  # purple-50
        ],
        "medium": [
            "#E5E7EB",  # gray-200
            "#BAE6FD",  # sky-200
            "#BBF7D0",  # green-200
            "#FED7AA",  # orange-200
            "#C7D2FE",  # indigo-200
            "#E9D5FF",  # purple-200
        ],
        "extreme": [
            "#D1D5DB",  # gray-300
            "#7DD3FC",  # sky-300
            "#86EFAC",  # green-300
            "#FDBA74",  # orange-300
            "#A5B4FC",  # indigo-300
            "#D8B4FE",  # purple-300
        ]
    }


    def __init__(self, experiment_name="", use_ema=False, ema_decay=0.9):
        """
        experiment_name : name of the experiment (shown in plots)
        use_ema         : apply EMA smoothing
        ema_decay       : EMA decay factor
        """
        self.experiment_name = experiment_name

        self.history = defaultdict(list)
        self.ema_history = defaultdict(list)

        self.epochs = []
        self.criteria = {}

        self.use_ema = use_ema
        self.ema_decay = ema_decay

        self.best_records = {}
        self.sub_records = None

        self.phase_records = None

    # ---------- core utils ----------

    def add_criteria(self, name, higher_is_better=True):
        self.criteria[name] = higher_is_better

    def _ema_update(self, key, value):
        if len(self.ema_history[key]) == 0:
            ema = value
        else:
            prev = self.ema_history[key][-1]
            ema = self.ema_decay * prev + (1 - self.ema_decay) * value
        self.ema_history[key].append(ema)

    def get_sub_candidates(self):
        return self.sub_candidates
    
    def set_sub_records(self, _sub_records):
        self.sub_records = _sub_records

    # ---------- logging ----------

    def log(self, epoch, prefix=None, **metrics):
        """
        Example:
            log(1, prefix="train", loss=0.5, auc=0.8)
        """
        if epoch not in self.epochs:
            self.epochs.append(epoch)

        for name, value in metrics.items():
            key = f"{prefix}_{name}" if prefix else name
            value = float(value)

            self.history[key].append(value)

            if self.use_ema:
                self._ema_update(key, value)

            base = name 
            if base not in self.auc_candidates:
                continue

            if base in self.criteria:
                hib = self.criteria[base]
                record = self.best_records.setdefault(
                    key, {"epoch": None, "value": None}
                )

                is_better = (
                    record["value"] is None or
                    (hib and value > record["value"]) or
                    (not hib and value < record["value"])
                )

                if is_better:
                    record["value"] = value
                    record["epoch"] = epoch

    def add_phase(self, phase_name, start_epoch, end_epoch):
        if self.phase_records is None:
            self.phase_records = []
            
        self.phase_records.append({
            "phase_name": phase_name,
            "start_epoch": start_epoch,
            "end_epoch": end_epoch,
        })

    def _update_best_records(self, save=True):
        """
        Detect best epoch/value for all metrics relative with auc and save to self.best_records.
        Backward-compatible for old experiments.
        """
        self.best_records = {}

        for metric, values in self.history.items():
            if len(values) == 0 or metric not in self.auc_candidates:
                continue

            # determine rule
            base = metric.split("_", 1)[-1]
            higher_is_better = self.criteria.get(base, True)

            values = np.asarray(values)
            idx = int(np.argmax(values) if higher_is_better else np.argmin(values))

            self.best_records[metric] = {
                "epoch": self.epochs[idx],
                "value": float(values[idx]),
            }

        # ---------- persist ----------
        if save:
            path = os.path.join(self.experiment_root, self.experiment_name, self.run_log_name)
            self.export_json(path)

    def _update_sub_records(self, ema_enable=False, save=True, path=None):
        """
        Compute and cache sub_records at best AUC epoch
        """

        # ---------- ensure best_records ----------
        if (
            not hasattr(self, "best_records")
            or not isinstance(self.best_records, dict)
            or len(self.best_records) == 0
        ):
            self._update_best_records(save=False)

        # ---------- find best AUC ----------
        best_epoch = None
        best_auc_key = None

        for k in self.auc_candidates:
            if k in self.best_records:
                rec = self.best_records[k]
                if rec.get("epoch") is not None:
                    best_epoch = rec["epoch"]
                    best_auc_key = k
                    break

        if best_epoch is None:
            return None

        try:
            best_idx = self.epochs.index(best_epoch)
        except ValueError:
            return None

        src = self.ema_history if ema_enable else self.history

        metrics = {}
        for k in self.sub_candidates:
            if k in src and best_idx < len(src[k]):
                metrics[k] = src[k][best_idx]
            else:
                metrics[k] = None

        self.sub_records = {
            "epoch": best_epoch,
            "auc_key": best_auc_key,
            "metrics": metrics,
        }

        # ---------- optional save ----------
        if save:
            if path is None:
                path = os.path.join(self.experiment_root, self.experiment_name, self.run_log_name)
            self.export_json(path)

        return self.sub_records


    # ---------- export / import ----------

    def export_json(self, path):
        payload = {
            "experiment_name": self.experiment_name,
            "epochs": self.epochs,
            "history": dict(self.history),
            "ema_history": dict(self.ema_history),
            "criteria": self.criteria,
            "best_records": self.best_records, 
            "sub_records": self.sub_records, 
            "phase_records": self.phase_records,
            "use_ema": self.use_ema,
            "ema_decay": self.ema_decay,
        }

        with open(path, "w") as f:
            json.dump(payload, f, indent=4)

        print(f"[TrainVisualizer] Exported to {path}")


    @classmethod
    def from_json(cls, path):
        with open(path, "r") as f:
            payload = json.load(f)

        obj = cls(
            experiment_name=payload.get("experiment_name", ""),
            use_ema=payload.get("use_ema", False),
            ema_decay=payload.get("ema_decay", 0.9),
        )

        obj.epochs = payload.get("epochs", [])
        obj.history = defaultdict(list, payload.get("history", {}))
        obj.ema_history = defaultdict(list, payload.get("ema_history", {}))
        obj.criteria = payload.get("criteria", {})
        obj.best_records = payload.get("best_records", {})
        obj.sub_records = payload.get("sub_records", None)
        obj.phase_records = payload.get("phase_records", None)

        return obj

    # ---------- visualization ----------

    def _to_roman(self, n: int) -> str:
        vals = [
            (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
            (100, "C"),  (90, "XC"),  (50, "L"),  (40, "XL"),
            (10, "X"),   (9, "IX"),   (5, "V"),   (4, "IV"),
            (1, "I"),
        ]

        res = []
        for v, s in vals:
            while n >= v:
                res.append(s)
                n -= v
        return "".join(res)


    def _draw(self, save_path=None, figsize=(6, 4), ema_enable=False):

        # ---------- group by base metric ----------
        groups = defaultdict(list)
        for key in self.history.keys():
            if "_" in key:
                _, base = key.split("_", 1)
            else:
                base = key
            groups[base].append(key)

        metrics = list(groups.items())
        n = len(metrics)

        # ---------- layout ----------
        n_cols = math.ceil(math.sqrt(n))
        n_rows = math.ceil(n / n_cols)

        if self.phase_records is not None:
            fig = plt.figure(
                figsize=(figsize[0] * n_cols, figsize[1] * (n_rows + 0.25))
            )

            gs = fig.add_gridspec(
                n_rows + 1,
                n_cols,
                height_ratios=[0.24] + [1.0] * n_rows
            )

            phase_ax = fig.add_subplot(gs[0, :])

            # metric axes
            axes = []
            for i in range(n):
                r = i // n_cols + 1
                c = i % n_cols
                axes.append(fig.add_subplot(gs[r, c]))

            phase_ax.set_xlim(min(self.epochs), max(self.epochs))
            phase_ax.set_ylim(0, 1)
            phase_ax.set_yticks([])
            phase_ax.set_xlabel("Epoch")
            phase_ax.set_ylabel("Phase\nName")

            for spine in ["top", "left", "right"]:
                phase_ax.spines[spine].set_visible(False)

            phase_ax.tick_params(axis="x", labelsize=9)

        else:
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(figsize[0] * n_cols, figsize[1] * n_rows)
            )

            if n_rows * n_cols == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

        # ---------- ensure best_records ----------
        if (
            not hasattr(self, "best_records")
            or self.best_records is None
            or (isinstance(self.best_records, dict) and len(self.best_records) == 0)
        ):
            self._update_best_records()

        # ---------- find best AUC ----------
        self.best_epoch = None
        self.best_auc = None
        self.best_auc_key = None

        for k in self.auc_candidates:
            if k in self.best_records:
                rec = self.best_records[k]
                if rec["epoch"] is None:
                    continue

                self.best_epoch = rec["epoch"]
                self.best_auc = rec["value"]
                self.best_auc_key = k
                break

        # ---------- phase legend ----------
        if self.phase_records and self.best_epoch is not None:

            for pid, phase in enumerate(self.phase_records):

                start = phase["start_epoch"]
                end   = phase["end_epoch"] + 1

                if start is None or end is None:
                    continue

                color = self.PHASE_COLORS[self.PHASE_KEY][pid % len(self.PHASE_COLORS[self.PHASE_KEY])]

                # ---- highlight logic ----
                is_best_phase = (start <= self.best_epoch <= end)

                linewidth = 1.8 if is_best_phase else 0.8
                alpha     = 0.50 if is_best_phase else 0.30
                zorder    = 3 if is_best_phase else 1

                # ---- phase bar ----
                phase_ax.axvspan(
                    start,
                    end,
                    ymin=0.25,
                    ymax=0.75,
                    facecolor=color,
                    alpha=alpha,
                    edgecolor="black",
                    linewidth=linewidth,
                    zorder=zorder,
                )

                # ---- phase label ----
                mid = (start + end) / 2
                roman = self._to_roman(pid + 1)

                label = f"Phase {roman}: {phase['phase_name']}"
                if is_best_phase:
                    label = "★ " + label

                phase_ax.text(
                    mid,
                    0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9 if not is_best_phase else 10,
                    fontweight="bold",
                    zorder=zorder + 1,
                )

        # ---------- plot ----------
        for ax, (base, keys) in zip(axes, metrics):

            # ---------- phase background ----------
            if hasattr(self, "phase_records") and self.phase_records:
                for pid, phase in enumerate(self.phase_records):

                    start = phase["start_epoch"]
                    stop = phase["end_epoch"] + 1

                    # skip unfinished phase
                    if start is None or stop is None:
                        continue

                    color = self.PHASE_COLORS[self.PHASE_KEY][pid % len(self.PHASE_COLORS[self.PHASE_KEY])]

                    # background span
                    ax.axvspan(
                        start,
                        stop,
                        color=color,
                        alpha=0.35,
                        zorder=0
                    )


            for k in keys:
                if ema_enable and k in self.ema_history:
                    values = self.ema_history[k]
                    label = f"{k} (EMA)"
                else:
                    values = self.history[k]
                    label = k

                ax.plot(self.epochs, values, label=label)

            # ---- best epoch visualization ----
            if self.best_epoch is not None:
                ax.axvline(
                    x=self.best_epoch,
                    linestyle="--",
                    color="gray",
                    alpha=0.7
                )

                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                margin_x = 0.015 * (xmax - xmin)
                margin_y = 0.03 * (ymax - ymin)

                ax.text(
                    self.best_epoch - margin_x,
                    ymin + margin_y,
                    f"epoch={self.best_epoch}",
                    rotation=90,
                    va="bottom",
                    ha="right",
                    fontsize=9,
                    color="gray"
                )

                # ---------- scatter & annotate metrics ----------
                best_idx = self.epochs.index(self.best_epoch)

                for k in keys:

                    is_auc = (k == self.best_auc_key)
                    is_sub = (k in self.sub_candidates)

                    if not (is_auc or is_sub):
                        continue

                    # ---- get value ----
                    if is_auc:
                        val = self.best_auc
                        label = f"{k}: {val:.4f}"
                        fontsize = 10
                        weight = "bold"
                    else:
                        src = self.ema_history if (ema_enable and k in self.ema_history) else self.history
                        val = src[k][best_idx]
                        label = f"{k}: {val:.4f}"
                        fontsize = 9
                        weight = "normal"

                    # ---- draw ----
                    ax.scatter(
                        self.best_epoch,
                        val,
                        s=55 if is_sub else 65,
                        zorder=5,
                        edgecolors="black",
                        linewidths=0.7
                    )

                    # ---------- adaptive text alignment ----------
                    xmin, xmax = ax.get_xlim()
                    x_range = xmax - xmin

                    # best_epoch position ratio in [0, 1]
                    x_ratio = (self.best_epoch - xmin) / x_range

                    # threshold: gần mép phải
                    RIGHT_EDGE_THRESH = 0.72

                    if x_ratio > RIGHT_EDGE_THRESH:
                        text_x = self.best_epoch - margin_x
                        ha = "right"
                    else:
                        text_x = self.best_epoch + margin_x
                        ha = "left"

                    ax.text(
                        text_x,
                        val,
                        label,
                        fontsize=fontsize,
                        fontweight=weight,
                        ha=ha,
                        va="bottom",
                        clip_on=True,   
                    )

            ax.set_title(base)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True)

        # ---------- turn off unused axes ----------
        for ax in axes[len(metrics):]:
            ax.axis("off")

        # ---------- experiment title ----------
        if self.experiment_name:
            fig.suptitle(
                self.experiment_name,
                fontsize=14,
                fontweight="bold"
            )

        # plt.tight_layout(rect=[0, 0, 1, 0.93])
        # plt.tight_layout(pad=1.2)

        fig.subplots_adjust(
            left=0.06,
            right=0.98,
            top=0.93,
            bottom=0.06,
            hspace=0.35,
            wspace=0.25
        )


        if save_path:
            plt.savefig(save_path, dpi=200)
            plt.close(fig)
        else:
            plt.show()


    def plot(self, figsize=(6, 4), ema_enable=False):
        """
        Show plots interactively
        """
        self._draw(
            save_path=None,
            figsize=figsize,
            ema_enable=ema_enable
        )


    def save_plot(self, path, figsize=(6, 4), ema_enable=False):
        """
        Save plots to PNG
        """
        if not str(path).endswith(".png"):
            path += ".png"

        self._draw(
            save_path=path,
            figsize=figsize,
            ema_enable=ema_enable
        )

        print(f"[TrainVisualizer] Plot saved to {path}")



    # ---------- summary ----------

    def summary(self):
        print("=== Training Summary ===")

        # ---------- metric-wise best ----------
        for base, hib in self.criteria.items():
            candidates = [k for k in self.history if k.endswith(base)]
            for k in candidates:
                values = self.history[k]
                best = max(values) if hib else min(values)
                print(f"{k}: best = {best:.4f}")

        # ---------- best epoch (global, AUC-based) ----------
        if (
            hasattr(self, "best_epoch")
            and self.best_epoch is not None
            and hasattr(self, "best_auc")
            and self.best_auc is not None
        ):
            print(
                "\n"
                f"[BEST EPOCH] epoch={self.best_epoch} "
                f"({self.best_auc_key} = {self.best_auc:.4f})"
            )

        # ---------- phase summary ----------
        if hasattr(self, "phase_records") and self.phase_records:
            print("\n=== Phase Summary ===")

            # find which AUC key is used
            auc_key = None
            for k in getattr(self, "auc_candidates", []):
                if k in self.history:
                    auc_key = k
                    break

            if auc_key is None:
                print("[WARN] No AUC metric found for phase analysis.")
                return

            auc_values = self.history[auc_key]

            for phase in self.phase_records:
                name = phase.get("phase_name", "Unnamed")
                s = phase.get("start_epoch")
                e = phase.get("end_epoch")

                if s is None or e is None:
                    continue

                # epoch indices
                idxs = [
                    i for i, ep in enumerate(self.epochs)
                    if s <= ep <= e
                ]

                if not idxs:
                    print(f"- {name}: epochs [{s}, {e}] → no data")
                    continue

                phase_aucs = [auc_values[i] for i in idxs]
                best_idx = idxs[int(np.argmax(phase_aucs))]
                best_ep = self.epochs[best_idx]
                best_val = auc_values[best_idx]

                print(
                    f"- {name:<20s} "
                    f"epochs [{s:>3d} → {e:<3d}] | "
                    f"best {auc_key} = {best_val:.4f} @ epoch {best_ep}"
                )


    def summarize_best_epoch(self, ema_enable=False, save=True, path=None):
        """
        Return cached sub_records if available,
        otherwise compute and cache.
        """

        if (
            hasattr(self, "sub_records")
            and isinstance(self.sub_records, dict)
            and "epoch" in self.sub_records
            and "metrics" in self.sub_records
        ):
            return self.sub_records

        return self._update_sub_records(
            ema_enable=ema_enable,
            save=save,
            path=path
        )



import argparse

def build_argparser():
    parser = argparse.ArgumentParser("TrainVisualizer")

    # ---------- experiment scope ----------
    parser.add_argument(
        "--experiment_root", "-root",
        type=str,
        default="./results",
        help="Root directory containing all experiments"
    )

    parser.add_argument(
        "--experiment_name", "-name",
        type=str,
        default=None,
        help="Name of a single experiment folder"
    )

    # ---------- logging ----------
    parser.add_argument(
        "--run_logging", "-log",
        type=str,
        default="run_logging.json",
        help="Training log json filename (relative to experiment folder)"
    )

    # ---------- plotting ----------
    parser.add_argument(
        "--plot_savepath",
        type=str,
        default="run_logging.png",
        help="Plot output filename (.png), relative to experiment folder"
    )

    parser.add_argument(
        "--ema_enable",
        action="store_true",
        help="Enable EMA curve when plotting"
    )

    # ---------- summary mode ----------
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help=(
            "Regex filter for experiment names when printing summary table.\n\n"
            "Usage examples:\n"
            "  OR  (match any):\n"
            "    --summary \"gate|aux|clinical\"\n\n"
            "  AND (match all, using lookahead):\n"
            "    --summary \"(?=.*gate)(?=.*aux)\"\n\n"
            "  NOT (exclude keyword):\n"
            "    --summary \"^(?!.*debug).*\"\n\n"
            "  Combination (AND + NOT):\n"
            "    --summary \"(?=.*gate)(?=.*aux)^(?!.*old).*\"\n\n"
            "Notes:\n"
            "  - Matching is applied on experiment folder names\n"
            "  - Regex is Python-compatible (re.search)\n"
            "  - Useful for ablation studies & experiment comparison"
        )
    )

    parser.add_argument(
        "--sort_by",
        type=str,
        default="auc",
        choices=["auc", "length"],
        help="Sort summary table by: 'auc' or 'length'"
    )

    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort in descending order (default for AUC)"
    )

    return parser


import os
import re

def find_experiments_by_regex(root, pattern):
    regex = re.compile(pattern)
    exps = []

    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path) and regex.search(name):
            exps.append(name)

    return sorted(exps)

def summarize_experiments(exp_root, exp_names, log_name, ema_enable=False):
    rows = []

    for name in exp_names:
        log_path = os.path.join(exp_root, name, log_name)
        if not os.path.exists(log_path):
            continue

        logger = TrainVisualizer.from_json(log_path)

        sub = logger.summarize_best_epoch(
            ema_enable=ema_enable,
            save=True,
            path=log_path
        )

        if sub is None:
            continue

        row = {
            "exp_name": name,
            "epoch": sub["epoch"],
            "auc": logger.best_records[sub["auc_key"]]["value"],
            **sub["metrics"],
        }

        rows.append(row)

    return rows

# ---------- print table ----------
TABLE_LAYOUT = {
    # main
    "exp_name": {"title": "Experiment", "width": 128, "type": "str"},
    "auc":        {"title": "Best ROC-AUC",   "width": 12,  "type": "float"},
    "epoch":      {"title": "Epoch",      "width": 8,   "type": "int"},

    # sub
    "val_pr_auc": {"title": "PR-AUC",   "width": 12,  "type": "float"},
    "val_sensitivity": {"title": "Sensitivity",   "width": 12,  "type": "float"},
    "val_specificity": {"title": "Specificity",   "width": 12,  "type": "float"},
    "val_f1": {"title": "F1-Score",   "width": 12,  "type": "float"},
}

def _truncate(s, width):
    if s is None:
        return "N/A"
    s = str(s)
    return s if len(s) <= width else s[: width - 3] + "..."


def _format_value(v, col_type):
    if v is None:
        return "N/A"
    if col_type == "float":
        return f"{v:.4f}"
    return str(v)


def print_summary_table(
    rows,
    layout=TABLE_LAYOUT,
    sort_by="auc",        # "auc" | "length" | any column key | None
    descending=True,
    auto_expand_sub=True, # <- NEW
):
    """
    rows:
        - list of dicts:
          {
              "exp_name": "...",
              "auc": 0.91,
              "epoch": 42,
              "val_f1": 0.63,
              "val_sensitivity": 0.71,
              ...
          }
    """

    if not rows:
        print("[WARN] No experiment results found.")
        return

    # ---------- normalize rows ----------
    norm_rows = []
    for r in rows:
        if isinstance(r, dict):
            norm_rows.append(r)
        else:
            name, auc, epoch = r
            norm_rows.append({
                "exp_name": name,
                "auc": auc,
                "epoch": epoch,
            })

    # ---------- auto expand layout for sub metrics ----------
    final_layout = dict(layout)

    if auto_expand_sub:
        for r in norm_rows:
            for k, v in r.items():
                if v is None:
                    continue
                if k not in final_layout:
                    # infer type
                    col_type = "float" if isinstance(v, (int, float)) else "str"
                    final_layout[k] = {
                        "title": k,
                        "width": 12,
                        "type": col_type,
                    }

    # ---------- sorting ----------
    if sort_by:
        if sort_by == "length":
            norm_rows = sorted(
                norm_rows,
                key=lambda r: len(r.get("exp_name", "")),
                reverse=descending
            )
        elif sort_by in final_layout:
            norm_rows = sorted(
                norm_rows,
                key=lambda r: (
                    r.get(sort_by)
                    if r.get(sort_by) is not None
                    else float("-inf")
                ),
                reverse=descending
            )

    # ---------- header ----------
    header = " | ".join(
        f"{cfg['title']:{cfg['width']}s}"
        for cfg in final_layout.values()
    )

    sep_len = sum(cfg["width"] for cfg in final_layout.values()) \
              + 3 * (len(final_layout) - 1)

    print("\n=== Summary Results ===")
    print(header)
    print("-" * sep_len)

    # ---------- rows ----------
    for r in norm_rows:
        cols = []
        for key, cfg in final_layout.items():
            val = r.get(key)
            val = _format_value(val, cfg.get("type"))
            val = _truncate(val, cfg["width"])
            cols.append(f"{val:{cfg['width']}s}")

        print(" | ".join(cols))



def main():
    parser = build_argparser()
    args = parser.parse_args()

    exp_root = args.experiment_root

    # ---------- summary mode ----------
    if args.summary:
        exp_names = find_experiments_by_regex(exp_root, args.summary)
        if not exp_names:
            print("[WARN] No experiments matched query")
            return

        rows = summarize_experiments(
            exp_root,
            exp_names,
            args.run_logging
        )

        # ---- smart default sort ----
        descending = (
            args.descending
            if args.sort_by != "auc"
            else True if not args.descending else True
        )

        print_summary_table(
            rows,
            sort_by=args.sort_by,
            descending=descending
        )
        return

    # ---------- single experiment ----------
    if args.experiment_name is None:
        print("[ERROR] --experiment_name is required unless --summary is used")
        return

    exp_dir = os.path.join(exp_root, args.experiment_name)
    run_logging_path = os.path.join(exp_dir, args.run_logging)

    print(f"[INFO] Reloading experiment from {run_logging_path}")

    logger = TrainVisualizer.from_json(run_logging_path)

    if args.plot_savepath:
        save_path = os.path.join(exp_dir, args.plot_savepath)
        if not save_path.endswith(".png"):
            save_path += ".png"

        logger.save_plot(save_path, ema_enable=args.ema_enable)
    else:
        logger.plot(ema_enable=args.ema_enable)

    logger.summary()


if __name__ == "__main__":
    main()
