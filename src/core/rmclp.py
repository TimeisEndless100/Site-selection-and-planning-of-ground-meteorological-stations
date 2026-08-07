from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_R_KM = 6371.0


def _haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2.astype(float))
    lon2_rad = np.radians(lon2.astype(float))
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _build_cover_lists(
    s_df: pd.DataFrame,
    d_df: pd.DataFrame,
    p_df: pd.DataFrame,
    radius_km: float,
) -> tuple[List[List[int]], np.ndarray, np.ndarray]:
    coords_s = np.radians(s_df[["lat", "lon"]].values)
    coords_d = np.radians(d_df[["lat", "lon"]].values)
    coords_p = np.radians(p_df[["lat", "lon"]].values) if len(p_df) > 0 else np.zeros((0, 2))
    w_d = d_df["demand"].values

    rad_cov = radius_km / EARTH_R_KM
    tree_s = BallTree(coords_s, metric="haversine")
    idxs_s_cover_d = tree_s.query_radius(coords_d, r=rad_cov)

    if len(p_df) > 0:
        tree_p = BallTree(coords_p, metric="haversine")
        idxs_p_cover_d = tree_p.query_radius(coords_d, r=rad_cov)
        covered = np.array([len(x) > 0 for x in idxs_p_cover_d])
    else:
        covered = np.zeros(len(d_df), dtype=bool)

    s_cover_d_list = [[] for _ in range(len(s_df))]
    for d_idx, s_indices in enumerate(idxs_s_cover_d):
        for s_idx in s_indices:
            s_cover_d_list[int(s_idx)].append(d_idx)
    return s_cover_d_list, covered, w_d


def _prepare_problem(
    infer_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    prob_thresh: float,
    rn_filter: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    infer = infer_df.copy()
    if "demand" in infer.columns:
        infer = infer.drop(columns=["demand"])
    df = infer.merge(demand_df[["pid", "demand"]], on="pid", how="left")
    if "demand" not in df.columns:
        if "demand_y" in df.columns:
            df["demand"] = df["demand_y"]
        elif "demand_x" in df.columns:
            df["demand"] = df["demand_x"]
        else:
            df["demand"] = 0.0
    df["demand"] = df["demand"].fillna(0)

    mask_prob = df["prob_mean"] >= prob_thresh
    mask_non_existing = mask_prob & (df["is_P"] == 0)
    mask_after_rn = mask_non_existing.copy()
    if rn_filter and "slope_max" in df.columns:
        mask_after_rn &= df["slope_max"] < 45

    s_df = df[mask_after_rn].reset_index(drop=True)
    d_df = df[df["demand"] > 1e-6].reset_index(drop=True)
    p_df = df[df["is_P"] == 1].reset_index(drop=True)

    filter_stats = {
        "total_nodes": int(len(df)),
        "after_prob_thresh": int(mask_prob.sum()),
        "after_non_existing_filter": int(mask_non_existing.sum()),
        "after_rn_filter": int(mask_after_rn.sum()),
        "candidates": int(len(s_df)),
        "demand_nodes": int(len(d_df)),
        "existing_sites": int(len(p_df)),
        "prob_thresh": float(prob_thresh),
        "rn_filter": bool(rn_filter),
    }
    return df, s_df, d_df, p_df, filter_stats


def rmclp_select_sites(
    infer_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    prob_thresh: float = 0.7,
    p_add: int = 100,
    radius_km: float = 5.0,
    rn_filter: bool = True,
    selection_mode: str = "rmclp",
    lambda_disp: float = 0.3,
) -> Dict[str, Any]:
    _, s_df, d_df, p_df, filter_stats = _prepare_problem(
        infer_df=infer_df,
        demand_df=demand_df,
        prob_thresh=prob_thresh,
        rn_filter=rn_filter,
    )
    filter_stats["selection_mode"] = str(selection_mode)
    filter_stats["lambda_disp"] = float(lambda_disp)

    coverage_curve: list[dict[str, Any]] = []
    if len(s_df) == 0 or len(d_df) == 0:
        coverage_curve.append(
            {
                "step": 0,
                "selected_pid": "",
                "marginal_gain": 0.0,
                "cumulative_cov": 0.0,
                "gain_pct": 0.0,
            }
        )
        return {
            "selected_sites": s_df.iloc[:0].copy(),
            "base_cov": 0.0,
            "final_cov": 0.0,
            "gain": 0.0,
            "gain_pct": 0.0,
            "candidates": int(len(s_df)),
        "selected": 0,
        "candidate_filter_stats": filter_stats,
        "coverage_curve": coverage_curve,
        }

    s_cover_d_list, covered, w_d = _build_cover_lists(
        s_df=s_df,
        d_df=d_df,
        p_df=p_df,
        radius_km=radius_km,
    )
    base_cov = float(w_d[covered].sum())
    selected_idx: list[int] = []
    selected_gain: list[float] = []
    selected_score: list[float] = []

    coverage_curve.append(
        {
            "step": 0,
            "selected_pid": "",
            "marginal_gain": 0.0,
            "selection_score": 0.0,
            "cumulative_cov": base_cov,
            "gain_pct": 0.0,
        }
    )

    for step in range(1, p_add + 1):
        candidate_rows: list[tuple[int, float, float]] = []
        for s_idx in range(len(s_df)):
            if s_idx in selected_idx:
                continue
            d_idxs = s_cover_d_list[s_idx]
            if not d_idxs:
                continue
            new_mask = ~covered[d_idxs]
            if not np.any(new_mask):
                continue
            gain = float(w_d[np.array(d_idxs)[new_mask]].sum())
            disp_km = 0.0
            if selection_mode == "rmclp" and selected_idx:
                lat = float(s_df.iloc[s_idx]["lat"])
                lon = float(s_df.iloc[s_idx]["lon"])
                sel_rows = s_df.iloc[selected_idx]
                dists = _haversine_km(lat, lon, sel_rows["lat"].values, sel_rows["lon"].values)
                disp_km = float(np.min(dists)) if len(dists) > 0 else 0.0
            candidate_rows.append((s_idx, gain, disp_km))

        if not candidate_rows:
            break

        if selection_mode == "mclp":
            best_idx, best_gain, _ = max(candidate_rows, key=lambda item: (item[1], item[2], -item[0]))
            best_score = best_gain
        elif selection_mode == "rmclp":
            gains = np.array([row[1] for row in candidate_rows], dtype=float)
            disp = np.array([row[2] for row in candidate_rows], dtype=float)
            gain_norm = gains / gains.max() if gains.max() > 0 else np.zeros_like(gains)
            disp_norm = disp / disp.max() if disp.max() > 0 else np.zeros_like(disp)
            scores = gain_norm + float(lambda_disp) * disp_norm
            best_pos = int(np.argmax(scores))
            best_idx, best_gain, _ = candidate_rows[best_pos]
            best_score = float(scores[best_pos])
        else:
            raise ValueError(f"Unsupported selection_mode: {selection_mode}")

        selected_idx.append(best_idx)
        selected_gain.append(best_gain)
        selected_score.append(best_score)
        covered[s_cover_d_list[best_idx]] = True
        cumulative_cov = float(w_d[covered].sum())
        gain_pct = float(((cumulative_cov - base_cov) / base_cov) * 100.0) if base_cov > 0 else 0.0
        coverage_curve.append(
            {
                "step": step,
                "selected_pid": str(s_df.iloc[best_idx]["pid"]) if "pid" in s_df.columns else str(best_idx),
                "marginal_gain": float(best_gain),
                "selection_score": float(best_score),
                "cumulative_cov": cumulative_cov,
                "gain_pct": gain_pct,
            }
        )

    final_cov = float(w_d[covered].sum())
    gain = float(final_cov - base_cov)
    gain_pct = float((gain / base_cov) * 100.0) if base_cov > 0 else 0.0

    selected_sites = s_df.iloc[selected_idx].copy()
    selected_sites["marginal_gain"] = selected_gain
    selected_sites["selection_score"] = selected_score
    return {
        "selected_sites": selected_sites,
        "base_cov": base_cov,
        "final_cov": final_cov,
        "gain": gain,
        "gain_pct": gain_pct,
        "candidates": int(len(s_df)),
        "selected": int(len(selected_idx)),
        "candidate_filter_stats": filter_stats,
        "coverage_curve": coverage_curve,
    }


def rmclp_gain(
    infer_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    prob_thresh: float = 0.7,
    p_add: int = 100,
    radius_km: float = 5.0,
    rn_filter: bool = True,
    selection_mode: str = "rmclp",
    lambda_disp: float = 0.3,
    return_details: bool = False,
) -> Dict[str, Any]:
    result = rmclp_select_sites(
        infer_df=infer_df,
        demand_df=demand_df,
        prob_thresh=prob_thresh,
        p_add=p_add,
        radius_km=radius_km,
        rn_filter=rn_filter,
        selection_mode=selection_mode,
        lambda_disp=lambda_disp,
    )
    out: Dict[str, Any] = {
        "base_cov": float(result["base_cov"]),
        "final_cov": float(result["final_cov"]),
        "gain": float(result["gain"]),
        "gain_pct": float(result["gain_pct"]),
        "candidates": int(result["candidates"]),
        "selected": int(result["selected"]),
    }
    if return_details:
        out["candidate_filter_stats"] = result.get("candidate_filter_stats", {})
        out["coverage_curve"] = result.get("coverage_curve", [])
    return out
