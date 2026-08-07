from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if len(rows) == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


class ExperimentLogger:
    def __init__(
        self,
        *,
        enabled: bool,
        log_dir: Path,
        exp_id: str,
        stage: str,
        model: str,
        seed: int,
        run_params: dict[str, Any],
        data_paths: dict[str, Any] | None = None,
    ) -> None:
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self.exp_id = exp_id
        self.stage = stage
        self.model = model
        self.seed = int(seed)
        self.start_time = time.time()
        self.started_at = _now_iso()
        self.fold_metrics_cache: list[dict[str, Any]] = []
        self._finalized = False

        if not self.enabled:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "exp_id": self.exp_id,
            "stage": self.stage,
            "model": self.model,
            "seed": self.seed,
            "started_at": self.started_at,
            "command": " ".join(sys.argv),
            "run_params": run_params,
            "data_paths": data_paths or {},
        }
        self.init_run(meta)

    @property
    def run_meta_path(self) -> Path:
        return self.log_dir / "run_meta.json"

    @property
    def split_path(self) -> Path:
        return self.log_dir / "data_split_summary.csv"

    @property
    def epoch_path(self) -> Path:
        return self.log_dir / "epoch_metrics.csv"

    @property
    def fold_path(self) -> Path:
        return self.log_dir / "fold_metrics.csv"

    @property
    def artifact_path(self) -> Path:
        return self.log_dir / "artifact_index.csv"

    @property
    def summary_path(self) -> Path:
        return self.log_dir / "run_summary.json"

    def init_run(self, meta: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.run_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def log_data_split(self, row: dict[str, Any] | list[dict[str, Any]]) -> None:
        if not self.enabled:
            return
        rows = row if isinstance(row, list) else [row]
        payload = []
        for r in rows:
            x = {"exp_id": self.exp_id, "stage": self.stage, "model": self.model, "seed": self.seed}
            x.update(r)
            payload.append(x)
        _append_csv(self.split_path, payload)

    def log_epoch(self, **kwargs: Any) -> None:
        if not self.enabled:
            return
        row = {
            "exp_id": self.exp_id,
            "stage": self.stage,
            "model": self.model,
            "seed": self.seed,
            "fold": kwargs.get("fold", -1),
            "epoch": kwargs.get("epoch", -1),
            "phase": kwargs.get("phase", ""),
            "loss": kwargs.get("loss", np.nan),
            # loss_pu / loss_rn 分量：支撑 λ_rn 量级论证（sens_lambda）与目标消融
            "loss_pu": kwargs.get("loss_pu", np.nan),
            "loss_rn": kwargs.get("loss_rn", np.nan),
            "lr": kwargs.get("lr", np.nan),
            "pi": kwargs.get("pi", np.nan),
            "lambda_rn": kwargs.get("lambda_rn", np.nan),
            "p_score": kwargs.get("p_score", np.nan),
            "rn_score": kwargs.get("rn_score", np.nan),
            "time_sec": kwargs.get("time_sec", np.nan),
        }
        _append_csv(self.epoch_path, [row])

    def log_fold_summary(self, **kwargs: Any) -> None:
        if not self.enabled:
            return
        row = {
            "exp_id": self.exp_id,
            "stage": self.stage,
            "model": self.model,
            "seed": self.seed,
            "fold": kwargs.get("fold", -1),
            "auc": kwargs.get("auc", np.nan),
            "auprc": kwargs.get("auprc", np.nan),
            "brier": kwargs.get("brier", np.nan),
            "sep_prn": kwargs.get("sep_prn", np.nan),
            "recall_k": kwargs.get("recall_k", np.nan),
            "runtime_min": kwargs.get("runtime_min", np.nan),
        }
        _append_csv(self.fold_path, [row])
        self.fold_metrics_cache.append(row)

    def log_artifact(
        self,
        *,
        path: str | Path,
        artifact_type: str,
        fold: int = -1,
        note: str = "",
    ) -> None:
        if not self.enabled:
            return
        row = {
            "exp_id": self.exp_id,
            "stage": self.stage,
            "model": self.model,
            "seed": self.seed,
            "fold": fold,
            "artifact_type": artifact_type,
            "path": str(path),
            "note": note,
            "logged_at": _now_iso(),
        }
        _append_csv(self.artifact_path, [row])

    def finalize_run(self, *, status: str, message: str = "") -> None:
        if not self.enabled or self._finalized:
            return
        self._finalized = True
        elapsed = time.time() - self.start_time
        summary: dict[str, Any] = {
            "exp_id": self.exp_id,
            "stage": self.stage,
            "model": self.model,
            "seed": self.seed,
            "status": status,
            "message": message,
            "started_at": self.started_at,
            "finished_at": _now_iso(),
            "duration_sec": elapsed,
        }
        if len(self.fold_metrics_cache) > 0:
            df = pd.DataFrame(self.fold_metrics_cache)
            for k in ["auc", "auprc", "brier", "sep_prn", "recall_k", "runtime_min"]:
                vals = df[k].dropna().values.astype(float)
                summary[f"{k}_mean"] = float(np.mean(vals)) if len(vals) > 0 else np.nan
        self.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def save_roc_pr_points(y_true: np.ndarray, y_score: np.ndarray, out_dir: Path, fold: int) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, roc_th = roc_curve(y_true, y_score)
    prec, rec, pr_th = precision_recall_curve(y_true, y_score)

    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_th})
    pr_threshold = np.full(len(prec), np.nan)
    if len(pr_th) > 0:
        pr_threshold[: len(pr_th)] = pr_th
    pr_df = pd.DataFrame({"precision": prec, "recall": rec, "threshold": pr_threshold})

    roc_path = out_dir / f"roc_points_fold_{fold}.csv"
    pr_path = out_dir / f"pr_points_fold_{fold}.csv"
    roc_df.to_csv(roc_path, index=False)
    pr_df.to_csv(pr_path, index=False)
    return {"roc": roc_path, "pr": pr_path}


def save_score_bins(
    *,
    set_values: np.ndarray,
    y_score: np.ndarray,
    out_dir: Path,
    fold: int,
    bins: int = 10,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    for group in ["P", "U", "RN"]:
        mask = set_values == group
        vals = y_score[mask]
        hist, _ = np.histogram(vals, bins=edges)
        total = int(hist.sum())
        for i in range(len(hist)):
            rows.append(
                {
                    "group": group,
                    "bin_left": float(edges[i]),
                    "bin_right": float(edges[i + 1]),
                    "count": int(hist[i]),
                    "fraction": float(hist[i] / total) if total > 0 else 0.0,
                }
            )
    path = out_dir / f"score_bins_fold_{fold}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def build_inference_summary(df: pd.DataFrame, high_prob_thresh: float = 0.8) -> dict[str, Any]:
    row = {
        "n_nodes": int(len(df)),
        "prob_mean_mean": float(df["prob_mean"].mean()),
        "prob_mean_std": float(df["prob_mean"].std(ddof=0)),
        "prob_std_mean": float(df["prob_std"].mean()) if "prob_std" in df.columns else np.nan,
        "high_prob_count": int((df["prob_mean"] >= high_prob_thresh).sum()),
    }
    if "is_P" in df.columns:
        row["high_prob_non_existing_count"] = int(((df["prob_mean"] >= high_prob_thresh) & (df["is_P"] == 0)).sum())
    else:
        row["high_prob_non_existing_count"] = np.nan
    return row

