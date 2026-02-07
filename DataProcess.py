import numpy as np
import pandas as pd

FLAG_Good = 0
FLAG_Erroneous = 1
FLAG_Suspect = 2
FLAG_Interpolated = 3


# ------------------------------------------------------------------------------
# Frequency determination and standardized timeline
# ------------------------------------------------------------------------------

def detect_and_convert(df, date_col="date", monthly_threshold=2):
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    out = out.dropna(subset=[date_col]).sort_values(date_col)
    out = out.groupby(date_col, as_index=False).mean(numeric_only=True)

    d = out[date_col]
    counts = d.groupby([d.dt.year, d.dt.month]).nunique()

    if (counts >= monthly_threshold).any():
        freq = "D"
    else:
        freq = "M"
        out[date_col] = out[date_col].dt.to_period("M").dt.to_timestamp(how="start")
        out = out.groupby(date_col, as_index=False).mean(numeric_only=True)

    out = out.sort_values(date_col).reset_index(drop=True)
    return freq, out


# ------------------------------------------------------------------------------
# Fill regularized time index within contiguous segments
# ------------------------------------------------------------------------------

def ensure_regular_time_index(df, date_col, freq, segment_break_d=60, segment_break_m=12):
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).set_index(date_col).sort_index()

    if freq.upper().startswith("D"):
        segment_break = segment_break_d
        gaps = out.index.to_series().diff().dt.days.fillna(0).astype(int)
        new_segment = gaps > segment_break

        def make_full_index(idx):
            return pd.date_range(idx.min().normalize(), idx.max().normalize(), freq="D", name=out.index.name)

    elif freq.upper().startswith("M"):
        segment_break = segment_break_m
        this_p = out.index.to_period("M")
        prev_p = this_p.shift(1)
        gaps = ((this_p.year - prev_p.year) * 12 + (this_p.month - prev_p.month)).fillna(0).astype(int)
        new_segment = gaps > segment_break

        def make_full_index(idx):
            start = pd.Period(idx.min(), freq="M").start_time
            end = pd.Period(idx.max(), freq="M").start_time
            return pd.date_range(start, end, freq="MS", name=out.index.name)
    else:
        raise ValueError("freq must be 'D' or 'M'")

    seg_id = new_segment.cumsum()
    dtypes = out.dtypes

    pieces = []
    for _, seg_df in out.groupby(seg_id, sort=True):
        full_idx = make_full_index(seg_df.index)
        seg_re = seg_df.reindex(full_idx, copy=False)

        for col, dtype in dtypes.items():
            if col in seg_re.columns:
                try:
                    seg_re[col] = seg_re[col].astype(dtype)
                except Exception:
                    pass
        pieces.append(seg_re)

    out2 = (
        pd.concat(pieces, axis=0)
        .sort_index()
        .reset_index()
        .rename(columns={"index": date_col})
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    return out2


# ------------------------------------------------------------------------------
# Adaptive robust Z (Median/MAD) — FLAG_Erroneous
# ------------------------------------------------------------------------------

def rolling_robust_z(x, win, min_periods):
    x = pd.to_numeric(x, errors="coerce")
    med = x.rolling(win, min_periods=min_periods).median()
    mad = (x - med).abs().rolling(win, min_periods=min_periods).median()

    med.fillna(x.median(), inplace=True)
    mad.fillna((x - x.median()).abs().median(), inplace=True)

    denom = (1.4826 * mad).replace(0, np.nan)
    z = ((x - med) / denom).abs()
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def flag_by_adaptive_z(g, col, flag_col, win=90, min_periods=90, q=0.99, multiplier=2):
    valid_mask = g[col].notna()
    zabs = rolling_robust_z(g.loc[valid_mask, col], win, min_periods)

    thr = zabs.quantile(q) * multiplier
    bad_mask = zabs > thr

    g.loc[zabs.index[bad_mask], flag_col] = FLAG_Erroneous
    g.loc[zabs.index[bad_mask], col] = np.nan
    g.loc[zabs.index, f"rb_{col}"] = zabs

    return g, thr


# ------------------------------------------------------------------------------
# Monotonic consistency between Level & Storage — FLAG_Suspect
# ------------------------------------------------------------------------------

def flag_storage_to_level(df, level_col="level", storage_col="storage", flag_level_col="flag_level", flag_storage_col="flag_storage"):
    out = df.copy()
    for col in [flag_level_col, flag_storage_col]:
        out[col] = out[col].fillna(0).astype(int)

    mask_ok = (out[flag_level_col].eq(0)) & (out[flag_storage_col].eq(0))
    valid = out.loc[mask_ok, [level_col, storage_col]].dropna()

    if valid.empty:
        return out

    sorted_vals = valid.sort_values(storage_col).reset_index()
    level_diff = sorted_vals[level_col].diff()

    for i in sorted_vals.index:
        if i == 0:
            continue
        if level_diff[i] < 0:
            idx_bad = sorted_vals.loc[i, "index"]
            prev_level = sorted_vals.loc[i - 1, level_col]
            cur_level = sorted_vals.loc[i, level_col]
            prev_storage = sorted_vals.loc[i - 1, storage_col]
            cur_storage = sorted_vals.loc[i, storage_col]

            if abs(cur_storage - prev_storage) < 1.0 and abs(cur_level - prev_level) > 5.0:
                out.loc[idx_bad, flag_level_col] = FLAG_Suspect

    return out


def flag_level_to_storage(df, level_col="level", storage_col="storage", flag_level_col="flag_level", flag_storage_col="flag_storage", abs_tol=0.1, rel_tol=0.002, side_k=5, require_both_sides=True):
    g = df.copy()
    mask_ok = g[level_col].notna() & g[storage_col].notna() & g[flag_storage_col].eq(FLAG_Good) & g[flag_level_col].eq(FLAG_Good)
    if mask_ok.sum() <= 2:
        return g

    tmp = g.loc[mask_ok, [level_col, storage_col]].copy()
    order = tmp[level_col].argsort(kind="mergesort").to_numpy()
    idx_sorted = tmp.index.to_numpy()[order]
    L = tmp[level_col].to_numpy()[order].astype(float)
    S = tmp[storage_col].to_numpy()[order].astype(float)
    n = S.size

    bad_sorted = np.zeros(n, dtype=bool)

    for i in range(n):
        left_vals = S[max(0, i - side_k):i]
        right_vals = S[i + 1:min(n, i + 1 + side_k)]

        if left_vals.size == 0 or right_vals.size == 0:
            continue

        left_med = np.median(left_vals[np.isfinite(left_vals)])
        right_med = np.median(right_vals[np.isfinite(right_vals)])

        dl = abs(S[i] - left_med)
        dr = abs(S[i] - right_med)
        thr_l = abs_tol + rel_tol * max(left_med, 0.0)
        thr_r = abs_tol + rel_tol * max(right_med, 0.0)

        cond_l = dl > thr_l
        cond_r = dr > thr_r

        bad_sorted[i] = (cond_l and cond_r) if require_both_sides else (cond_l or cond_r)

    if bad_sorted.any():
        g.loc[idx_sorted[bad_sorted], flag_storage_col] = FLAG_Suspect

    return g


# ------------------------------------------------------------------------------
# Consecutive identical values ——FLAG_Suspect
# ------------------------------------------------------------------------------

def flag_by_consecutive_equals(g, col, flag_col, min_run=5):
    s = g[col]
    valid = s.notna()
    s_valid = s[valid]

    grp_ids = (s_valid != s_valid.shift()).cumsum()
    runlen = grp_ids.groupby(grp_ids).transform('size')

    mask_valid = runlen >= min_run
    idx_flag = s_valid.index[mask_valid]
    g.loc[idx_flag, flag_col] = FLAG_Suspect

    return g


# ------------------------------------------------------------------------------
# Cross-variable completion thought monotonic mapping — FLAG_Interpolated
# ------------------------------------------------------------------------------

def fill_by_counterpart(df, level_col="level", storage_col="storage", flag_level_col="flag_level", flag_storage_col="flag_storage"):
    g = df.copy()
    clean_both = g[flag_level_col].eq(FLAG_Good) & g[flag_storage_col].eq(FLAG_Good)

    base_L = g.loc[clean_both, [level_col, storage_col]].dropna()
    s2l = base_L.groupby(storage_col)[level_col].median().sort_index()
    base_S = g.loc[clean_both, [level_col, storage_col]].dropna()
    l2s = base_S.groupby(level_col)[storage_col].median().sort_index()

    def _round_layers_lookup(val, src, tgt, okmask):
        for rd in (2, 1, 0):
            m = okmask & g[src].notna() & g[tgt].notna() & (g[src].round(rd) == round(val, rd))
            if m.any():
                return g.loc[m, tgt].median()
        return np.nan

    def _bracket_average(series_map: pd.Series, val: float):
        if pd.isna(val) or series_map.empty:
            return np.nan
        ix = series_map.index.to_numpy(dtype=float)
        pos = np.searchsorted(ix, val, side="left")

        left_v = series_map.loc[ix[pos - 1]] if pos > 0 else np.nan
        right_v = series_map.loc[ix[pos]] if pos < ix.size else np.nan

        if np.isfinite(left_v) and np.isfinite(right_v):
            return 0.5 * (left_v + right_v)
        return left_v if np.isfinite(left_v) else (right_v if np.isfinite(right_v) else np.nan)

    def _find_level_from_storage(s):
        if s in s2l.index:
            return s2l.loc[s]
        v = _round_layers_lookup(s, storage_col, level_col, g[flag_storage_col].eq(FLAG_Good))
        return v if pd.notna(v) else _bracket_average(s2l, float(s))

    def _find_storage_from_level(lv):
        if lv in l2s.index:
            return l2s.loc[lv]
        v = _round_layers_lookup(lv, level_col, storage_col, g[flag_level_col].eq(FLAG_Good))
        return v if pd.notna(v) else _bracket_average(l2s, float(lv))

    need_L = g[level_col].isna() & g[storage_col].notna()  # & g[flag_storage_col].eq(FLAG_Good)
    for i, row in g.loc[need_L].iterrows():
        nv = _find_level_from_storage(row[storage_col])
        if pd.notna(nv):
            g.at[i, level_col] = nv
            if g.at[i, flag_level_col] == FLAG_Good:
                g.at[i, flag_level_col] = FLAG_Interpolated

    need_S = g[storage_col].isna() & g[level_col].notna()  # & ~g[flag_level_col].eq(FLAG_Erroneous)
    for i, row in g.loc[need_S].iterrows():
        nv = _find_storage_from_level(row[level_col])
        if pd.notna(nv):
            g.at[i, storage_col] = nv
            if g.at[i, flag_storage_col] == FLAG_Good:
                g.at[i, flag_storage_col] = FLAG_Interpolated

    return g


# ------------------------------------------------------------------------------
# Infer missing level from storage — FLAG_Interpolated
# ------------------------------------------------------------------------------

def fill_level_from_storage(df, storage_col="storage", level_col="level", flag_level_col="flag_level", storage_tol=0.0):
    out = df.copy()
    out[flag_level_col] = out[flag_level_col].fillna(FLAG_Good).astype(int)

    known_mask = out[level_col].notna() & out[storage_col].notna()
    if known_mask.sum() == 0:
        return out

    known = out.loc[known_mask, [storage_col, level_col]]
    order = np.argsort(known[storage_col].to_numpy(dtype=float))
    ks = known[storage_col].to_numpy(dtype=float)[order]
    kl = known[level_col].to_numpy(dtype=float)[order]

    miss_mask = out[level_col].isna() & out[storage_col].notna()

    for idx in out.index[miss_mask]:
        s = float(out.at[idx, storage_col])

        exact_hits = np.isclose(ks, s, atol=storage_tol, rtol=0.0) if storage_tol > 0 else (ks == s)
        if np.any(exact_hits):
            vals = kl[exact_hits]
            n = vals.size
            fill_val = float(vals[0]) if n == 1 else (
                float((vals[0] + vals[1]) / 2.0) if n == 2 else float(np.median(vals)))
            out.at[idx, level_col] = fill_val
            continue

        pos = np.searchsorted(ks, s, side="left")
        left_ok = (pos - 1) >= 0
        right_ok = pos < ks.size

        if left_ok and right_ok:
            sL, lL = ks[pos - 1], kl[pos - 1]
            sR, lR = ks[pos], kl[pos]

            if not np.isclose(sL, sR):
                t = (s - sL) / (sR - sL)
                out.at[idx, level_col] = float(lL + t * (lR - lL))
                out.at[idx, flag_level_col] = FLAG_Interpolated
            else:
                out.at[idx, level_col] = float(lL if abs(s - sL) <= abs(s - sR) else lR)
        elif left_ok:
            out.at[idx, level_col] = float(kl[pos - 1])
        elif right_ok:
            out.at[idx, level_col] = float(kl[pos])

    return out


# ------------------------------------------------------------------------------
# Final time interpolation — FLAG_Interpolated
# ------------------------------------------------------------------------------

def final_time_interp(df, date_col="date", val_col="level", flag_col="flag_level", limit=3, fill_edges=False):
    g = df.copy()
    g[date_col] = pd.to_datetime(g[date_col], errors="coerce")
    g = g.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

    if isinstance(g.index, pd.PeriodIndex):
        g.index = g.index.to_timestamp(how="start")
    else:
        g.index = pd.to_datetime(g.index, errors="coerce")
        g = g[~g.index.isna()]
        if getattr(g.index, "tz", None) is not None:
            g.index = g.index.tz_localize(None)

    if val_col in g.columns:
        before = g[val_col].isna()

        g[val_col] = g[val_col].interpolate(method="linear", limit=limit, limit_area="inside")
        filled_inside = before & g[val_col].notna()
        m = filled_inside & ~g[flag_col].isin([FLAG_Erroneous, FLAG_Suspect])
        g.loc[m, flag_col] = FLAG_Interpolated

        if fill_edges:
            before_edges = g[val_col].isna()
            g[val_col] = g[val_col].interpolate(method="linear", limit=limit, limit_direction="forward")
            g[val_col] = g[val_col].interpolate(method="linear", limit=limit, limit_direction="backward")
            filled_edge = before_edges & g[val_col].notna()
            m = filled_edge & ~g[flag_col].isin([FLAG_Erroneous, FLAG_Suspect])
            g.loc[m, flag_col] = FLAG_Interpolated

    return g.reset_index().rename(columns={"index": date_col})


# ===========================================
# Single-variable QC
# ===========================================

def qc_level_or_storage(df, station_col, date_col, value_col):
    name = value_col.lower()
    if "level" in name:
        var, flag, raw = "level", "flag_level", "level_raw"
    elif "stor" in name:
        var, flag, raw = "storage", "flag_storage", "storage_raw"
    else:
        var, flag, raw = "level", "flag_level", "level_raw"

    rename_map = {station_col: "id", date_col: "date", value_col: var}
    g = df.rename(columns=rename_map).copy()[["id", "date", var]]

    id_val = g['id'].unique().item()
    g[var] = pd.to_numeric(g[var], errors="coerce")
    g[raw] = g[var]
    g[flag] = FLAG_Good
    g.loc[g[var] == 0, var] = np.nan

    frequency, g = detect_and_convert(g, "date")
    max_gap = 60 if str(frequency).upper().startswith('D') else 12
    win = 90 if str(frequency).upper().startswith('D') else 36

    if "Level" in value_col:
        g, thr = flag_by_adaptive_z(g, "level", "flag_level", win=win, min_periods=win)
    else:
        g, thr = flag_by_adaptive_z(g, "storage", "flag_storage", win=win, min_periods=win, multiplier=5.5)

    g.loc[g[flag].isin([FLAG_Erroneous, FLAG_Suspect]), var] = np.nan

    g = flag_by_consecutive_equals(g, var, flag)
    g.loc[g[flag].isin([FLAG_Erroneous, FLAG_Suspect]), var] = np.nan

    g = ensure_regular_time_index(g, "date", frequency)
    g[flag] = g[flag].fillna(FLAG_Interpolated)
    g = final_time_interp(g, val_col=var, flag_col=flag, limit=max_gap)

    g.loc[g[raw].isna() & g[var].notna(), flag] = FLAG_Interpolated
    g = g.dropna(subset=[var], how="all")

    g["id"] = id_val
    cols_out = ["id", "date", raw, flag, var]
    return g[cols_out]


# ===========================================
# Dual-variable QC (level & storage)
# ===========================================

def qc_level_and_storage(df, station_col, date_col, level_col, storage_col):
    rename_cols = {station_col: "id", date_col: "date", level_col: "level", storage_col: "storage"}
    g = df.rename(columns=rename_cols).copy()[["id", "date", "level", "storage"]]
    id_val = g['id'].unique().item()

    g['level'] = pd.to_numeric(g['level'], errors='coerce')
    g['storage'] = pd.to_numeric(g['storage'], errors='coerce')
    g["level_raw"] = g["level"]
    g["storage_raw"] = g["storage"]
    g["flag_level"] = FLAG_Good
    g["flag_storage"] = FLAG_Good

    g.loc[g["level"] == 0, "level"] = np.nan
    g.loc[g["storage"] == 0, "storage"] = np.nan

    g.loc[g["level_raw"] == 0, "flag_level"] = FLAG_Erroneous
    g.loc[g["level_raw"] == 0, "flag_storage"] = FLAG_Erroneous

    frequency, g = detect_and_convert(g, "date")
    max_gap = 60 if str(frequency).upper().startswith('D') else 12
    win = 90 if str(frequency).upper().startswith('D') else 36

    g, thr_l = flag_by_adaptive_z(g, "level", "flag_level", win=win, min_periods=win)
    g, thr_s = flag_by_adaptive_z(g, "storage", "flag_storage", win=win, min_periods=win, multiplier=5.5)
    # print(thr_l,thr_s)
    g.loc[g["flag_level"].isin([FLAG_Erroneous, FLAG_Suspect]), "level"] = np.nan
    g.loc[g["flag_storage"].isin([FLAG_Erroneous, FLAG_Suspect]), "storage"] = np.nan

    g = flag_storage_to_level(g, level_col="level", storage_col="storage", flag_level_col="flag_level", flag_storage_col="flag_storage")

    g.loc[g["flag_level"].isin([FLAG_Erroneous, FLAG_Suspect]), "level"] = np.nan
    g.loc[g["flag_storage"].isin([FLAG_Erroneous, FLAG_Suspect]), "storage"] = np.nan

    g = flag_by_consecutive_equals(g, "level", "flag_level")
    g = flag_by_consecutive_equals(g, "storage", "flag_storage")

    g.loc[g["flag_level"].isin([FLAG_Erroneous, FLAG_Suspect]), "level"] = np.nan
    g.loc[g["flag_storage"].isin([FLAG_Erroneous, FLAG_Suspect]), "storage"] = np.nan

    g = ensure_regular_time_index(g, "date", frequency)
    g["flag_level"] = g["flag_level"].fillna(FLAG_Good)
    g["flag_storage"] = g["flag_storage"].fillna(FLAG_Good)

    g = fill_by_counterpart(g)

    g = fill_level_from_storage(g, storage_col="storage", level_col="level", flag_level_col="flag_level", storage_tol=1e-6)

    g = final_time_interp(g, val_col="level", flag_col="flag_level", limit=max_gap)
    g = fill_by_counterpart(g)

    mL = g["level_raw"].isna() & g["level"].notna() & ~g["flag_level"].isin([FLAG_Erroneous, FLAG_Suspect])
    g.loc[mL, "flag_level"] = FLAG_Interpolated

    mS = g["storage_raw"].isna() & g["storage"].notna() & ~g["flag_storage"].isin([FLAG_Erroneous, FLAG_Suspect])
    g.loc[mS, "flag_storage"] = FLAG_Interpolated

    g = g.dropna(subset=["level", "storage"], how="all")
    g["id"] = id_val
    cols_out = ["id", "date", "level_raw", "flag_level", "level", "storage_raw", "flag_storage", "storage"]

    return g[cols_out]

