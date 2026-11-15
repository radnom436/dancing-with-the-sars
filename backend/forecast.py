"""
Surface-water pattern mining + forecasting from REAL NASA data.

Data sources (nothing else is used, nothing is fabricated)
----------------------------------------------------------
* ASF DAAC          -> which Sentinel-1 dates/footprints exist over the site
                       (asf_client.site_timeline, already in this backend)
* NASA GIBS (Earthdata) -> OPERA L3 DSWx-S1 surface-water maps and OPERA L2 RTC-S1
                       backscatter for each of those dates (gibs_client.fetch_gibs_png)

Pipeline
--------
1. Build a (time x row x col) stack of observed water / not-water / not-observed
   per pixel. "Not observed" (outside the swath, masked) is NEVER treated as dry.
2. Mine patterns per pixel: water frequency, seasonal amplitude + peak day,
   long-term trend with a significance test, and a permanent / seasonal /
   intermittent / dry classification.
3. Forecast: per-pixel regularised logistic regression on
   [1, trend, annual (and 2nd) harmonics], fitted by IRLS on the observed dates only.
   The model form is chosen by a hold-out back-test on the most recent real dates,
   against two baselines (climatology, persistence). Skill is reported honestly;
   `reliable=False` is returned when the model does not beat the baselines.
4. Output: pattern maps, observed frames and forecast frames (PNG data URLs) on one
   common grid, plus predicted water area with an uncertainty band.

The forecast is a statistical extrapolation of what NASA observed. It is labelled as such
everywhere and is never mixed into the observed frames.
"""
from __future__ import annotations

import base64
import io
import logging
import math
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date as _date, datetime, timedelta, timezone

import numpy as np
from fastapi import HTTPException
from PIL import Image, ImageDraw

from asf_client import site_timeline
from gibs_client import fetch_gibs_png

log = logging.getLogger("sars.forecast")

STEP_DAYS = 12            # Sentinel-1 nominal repeat -> forecast cadence
MIN_PIXEL_OBS = 6         # a pixel needs this many valid dates to be modelled
MIN_FRAME_COVER = 0.10    # drop dates with almost no valid pixels
YEAR = 365.25
EPOCH = _date(2000, 1, 1).toordinal()   # phase 0 = 1 Jan, so "peak day" is a calendar day-of-year

# Candidate model forms; chosen by back-test (harmonic ones need >= ~1 year of data).
SPECS = {
    "seasonal":       dict(harm=1, trend=False, lam=2.0),
    "seasonal+trend": dict(harm=1, trend=True, lam=2.0),
    "seasonal-2nd":   dict(harm=2, trend=False, lam=4.0),
    "seasonal-soft":  dict(harm=1, trend=False, lam=12.0),
}
DEFAULT_SPEC = "seasonal"


# --------------------------------------------------------------------------
# 1. Observations from GIBS
# --------------------------------------------------------------------------

def _decode(png: bytes, grid: int) -> np.ndarray:
    im = Image.open(io.BytesIO(png)).convert("RGBA")
    if im.size != (grid, grid):
        im = im.resize((grid, grid), Image.NEAREST)
    return np.asarray(im)


def classify_wtr(rgba: np.ndarray):
    """NASA DSWx-S1 legend colours -> (water, has_data, masked). blue = open water,
    green = inundated vegetation, any other opaque colour = masked (not usable)."""
    r, g, b, a = (rgba[..., i].astype(np.int16) for i in range(4))
    has = a >= 20
    open_w = has & (b >= r + 30) & (b >= g + 10)
    veg = has & ~open_w & (g >= r + 30) & (g >= b + 30)
    masked = has & ~open_w & ~veg
    return open_w | veg, has, masked


def footprint_mask(acqs: list, bbox: tuple, grid: int) -> np.ndarray:
    w, s, e, n = bbox
    im = Image.new("L", (grid, grid), 0)
    d = ImageDraw.Draw(im)
    for a in acqs:
        fp = a.get("footprint") or []
        if len(fp) >= 3:
            d.polygon([((lo - w) / (e - w) * grid, (n - la) / (n - s) * grid) for lo, la in fp], fill=1)
    return np.asarray(im, dtype=bool)


def make_observation(day: str, rtc: np.ndarray | None, wtr: np.ndarray | None, fp: np.ndarray) -> dict | None:
    """Combine the two NASA layers + the ASF footprint into one observation.

    * No water product that day -> no observation (radar alone cannot say wet/dry; it must never become "dry").
    * valid = inside the footprint, NASA has data there, and the pixel is not masked. When the RTC layer is
      missing, land pixels are transparent in the water layer, so validity falls back to the footprint."""
    if wtr is None:
        return None
    water, has_w, masked = classify_wtr(wtr)
    if rtc is not None:
        has_rtc = rtc[..., 3] >= 20
        valid = fp & (has_rtc | has_w) & ~masked
    else:
        valid = fp & ~masked
    water = water & valid
    lum = None
    if rtc is not None:
        rgb = rtc[..., :3].astype(np.float32)
        lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
        lum = np.where(has_rtc, lum, np.nan).astype(np.float32)
    return {"date": _date.fromisoformat(day), "valid": valid, "water": water, "lum": lum, "has_wtr": True}


def _get_layer(layer: str, bbox_s: str, day: str, grid: int):
    try:
        return _decode(fetch_gibs_png(layer, bbox_s, day, grid), grid)
    except HTTPException as exc:
        if exc.status_code == 404:
            return None  # NASA has no product that day: final, not an error
        raise


def collect_dates(site_id: str, bbox_s: str, lookback_days: int, max_frames: int):
    """Distinct Sentinel-1 dates (+ their footprints) from ASF, evenly thinned to max_frames."""
    now = datetime.now(timezone.utc)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    items, errors = [], {}
    end = now
    remaining = lookback_days
    while remaining > 0:                      # 1-year chunks: ASF caps results per query
        span = min(365, remaining)
        start = end - timedelta(days=span)
        try:
            tl = site_timeline(site_id, bbox_s, start.strftime(fmt), end.strftime(fmt))
            items += [a for a in tl["acquisitions"] if str(a.get("platform") or "").startswith("Sentinel-1")]
            errors.update(tl.get("errors") or {})
        except HTTPException as exc:
            raise HTTPException(502, f"ASF DAAC timeline failed: {exc.detail}")
        end, remaining = start, remaining - span
    by_day: dict = {}
    for a in items:
        by_day.setdefault(a["time"][:10], {})[a.get("id") or id(a)] = a
    days = sorted(by_day)
    total = len(days)
    if total > max_frames:
        idx = sorted(set(np.linspace(0, total - 1, max_frames).round().astype(int).tolist()))
        days = [days[i] for i in idx]
    return [(d, list(by_day[d].values())) for d in days], total, errors


def fetch_observations(site_id, bbox, grid, lookback_days, max_frames, progress=None):
    bbox_s = ",".join(str(v) for v in bbox)
    cand, total_sar, errors = collect_dates(site_id, bbox_s, lookback_days, max_frames)
    if not cand:
        raise HTTPException(404, "ASF DAAC lists no Sentinel-1 acquisitions over this site for the requested period.")
    stats = {"sar_dates_in_period": total_sar, "dates_requested": len(cand), "no_product": 0, "failed": 0,
             "asf_errors": errors}
    out, done = [], [0]
    lock = threading.Lock()

    def work(item):
        day, acqs = item
        try:
            wtr = _get_layer("wtr", bbox_s, day, grid)
            rtc = _get_layer("rtc", bbox_s, day, grid) if wtr is not None else None   # skip a useless request
        except HTTPException:
            with lock:
                stats["failed"] += 1
            return None
        finally:
            with lock:
                done[0] += 1
                if progress:
                    progress("Fetching NASA OPERA imagery", done[0], len(cand))
        ob = make_observation(day, rtc, wtr, footprint_mask(acqs, bbox, grid))
        if ob is None:
            with lock:
                stats["no_product"] += 1
        return ob

    with ThreadPoolExecutor(max_workers=6) as ex:
        for ob in ex.map(work, cand):
            if ob is not None:
                out.append(ob)
    if stats["failed"] > len(cand) / 2:
        raise HTTPException(502, "NASA GIBS is not responding reliably right now; please retry later.")
    out.sort(key=lambda o: o["date"])
    return out, stats


# --------------------------------------------------------------------------
# 2. Model: per-pixel regularised logistic regression (vectorised IRLS)
# --------------------------------------------------------------------------

def design(days: np.ndarray, spec: dict, dref: float) -> np.ndarray:
    d = np.asarray(days, float)
    ph = 2 * np.pi * d / YEAR
    cols = [np.ones_like(d)]
    if spec["trend"]:
        cols.append((d - dref) / YEAR)
    for h in range(1, spec["harm"] + 1):
        cols += [np.sin(h * ph), np.cos(h * ph)]
    return np.stack(cols, axis=1)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(X: np.ndarray, Y: np.ndarray, V: np.ndarray, lam: float, iters: int = 10,
                 chunk: int = 6000, want_se: bool = False):
    """X (T,F); Y,V (T,P) in {0,1}. Ridge on all but the intercept. Returns B (P,F) [, se (P,F)]."""
    T, F = X.shape
    P = Y.shape[1]
    B = np.zeros((P, F))
    SE = np.zeros((P, F)) if want_se else None
    pen = np.full(F, lam)
    pen[0] = 1e-2
    D = np.diag(pen)
    for s in range(0, P, chunk):
        y = Y[:, s:s + chunk].T.astype(np.float64)
        v = V[:, s:s + chunk].T.astype(np.float64)
        n = v.sum(1)
        p0 = ((y * v).sum(1) + 0.5) / (n + 1.0)
        b = np.zeros((y.shape[0], F))
        b[:, 0] = np.log(p0 / (1 - p0))
        for _ in range(iters):
            mu = _sigmoid(b @ X.T)
            w = v * mu * (1 - mu)
            g = ((y - mu) * v) @ X - pen * b
            H = np.einsum("pt,tf,tg->pfg", w, X, X, optimize=True) + D
            step = np.linalg.solve(H, g[..., None])[..., 0]
            b += np.clip(step, -4, 4)
        B[s:s + chunk] = b
        if want_se:
            mu = _sigmoid(b @ X.T)
            w = v * mu * (1 - mu)
            H = np.einsum("pt,tf,tg->pfg", w, X, X, optimize=True) + D
            SE[s:s + chunk] = np.sqrt(np.maximum(np.diagonal(np.linalg.inv(H), axis1=1, axis2=2), 0))
    return (B, SE) if want_se else B


def _predict(B, spec, days, dref):
    return _sigmoid(design(days, spec, dref) @ B.T)  # (M,P)


def _score(prob, truth, valid, domain_n):
    m = valid
    if not m.any():
        return None
    diff = prob - truth
    brier = float((diff[m] ** 2).mean())
    pred = prob >= 0.5
    t = truth.astype(bool)
    acc = float((pred == t)[m].mean())
    tp, fp, fn = (pred & t & m).sum(), (pred & ~t & m).sum(), (~pred & t & m).sum()
    csi = float(tp / (tp + fp + fn)) if (tp + fp + fn) else None
    per = [float(prob[k][m[k]].mean() - truth[k][m[k]].mean()) for k in range(len(m)) if m[k].sum() >= 0.3 * domain_n]
    rmse = float(np.sqrt(np.mean(np.square(per)))) if per else None
    return {"brier": brier, "accuracy": acc, "csi_water": csi, "area_rmse_frac": rmse}


def backtest(days, Y, V, domain):
    """Train on all but the most recent K real dates, test on those K. Returns (results, best_name)."""
    T = len(days)
    K = max(3, int(round(0.2 * T)))
    if T - K < 8:
        return None, DEFAULT_SPEC
    tr, te = slice(0, T - K), slice(T - K, T)
    dref = float(np.mean(days[tr]))
    span = days[tr][-1] - days[tr][0]
    Ytr, Vtr = Y[tr][:, domain], V[tr][:, domain]
    keep = Vtr.sum(0) >= MIN_PIXEL_OBS
    if keep.sum() < 50:
        return None, DEFAULT_SPEC
    Ytr, Vtr = Ytr[:, keep], Vtr[:, keep]
    Yte, Vte = Y[te][:, domain][:, keep].astype(bool), V[te][:, domain][:, keep].astype(bool)
    n = Vtr.sum(0)
    freq = (((Ytr * Vtr).sum(0)) + 0.5) / (n + 1.0)
    clim = np.tile(freq, (K, 1))
    last = np.full(freq.shape, np.nan)
    for t in range(Ytr.shape[0]):
        last = np.where(Vtr[t] > 0, Ytr[t], last)
    persist = np.tile(np.clip(np.where(np.isnan(last), freq, last), 0.05, 0.95), (K, 1))
    res = {"climatology": _score(clim, Yte, Vte, keep.sum()), "persistence": _score(persist, Yte, Vte, keep.sum())}
    if span >= 300:
        for name, spec in SPECS.items():
            B = fit_logistic(design(days[tr], spec, dref), Ytr, Vtr, spec["lam"])
            res[name] = _score(_predict(B, spec, days[te], dref), Yte, Vte, keep.sum())
    res = {k: v for k, v in res.items() if v}
    if not res:
        return None, DEFAULT_SPEC
    best = min(res, key=lambda k: res[k]["brier"])
    return {"n_train": int(T - K), "n_test": int(K), "models": res, "best": best,
            "span_train_days": int(span)}, best


# --------------------------------------------------------------------------
# 3. Rendering helpers
# --------------------------------------------------------------------------

def _data_url(rgba: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgba.astype(np.uint8), "RGBA").save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _context_png(obs, shape):
    """Grey backdrop = temporal median of NASA's RTC-S1 rendering (real SAR texture, speckle-free)."""
    stack = [o["lum"] for o in obs if o["lum"] is not None]
    rgba = np.zeros(shape + (4,), np.uint8)
    rgba[..., :3], rgba[..., 3] = 12, 255
    if len(stack) >= 3:
        import warnings
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)   # pixels never seen by RTC are all-NaN
            med = np.nanmedian(np.stack(stack), axis=0)
        ok = np.isfinite(med)
        if ok.sum() > 100:
            lo, hi = np.percentile(med[ok], [2, 98])
            g = np.nan_to_num(np.clip((med - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.85)
            v = (25 + 190 * g).astype(np.uint8)
            for c in range(3):
                rgba[..., c] = np.where(ok, v, 12)
    return _data_url(rgba)


def _observed_png(o):
    h, w = o["valid"].shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[~o["valid"]] = (0, 0, 0, 120)                # not observed this date (dim, never "dry")
    rgba[o["water"]] = (40, 140, 255, 235)
    return _data_url(rgba)


def _forecast_png(prob, domain_mask, shape):
    h, w = shape
    p = np.zeros(h * w)
    p[domain_mask] = prob
    p = p.reshape(h, w)
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[~domain_mask.reshape(h, w)] = (0, 0, 0, 120)
    a = np.clip((p - 0.10) / 0.9, 0, 1)
    wet = domain_mask.reshape(h, w) & (p >= 0.10)
    rgba[..., 0] = np.where(wet, 150 - 130 * a, rgba[..., 0])
    rgba[..., 1] = np.where(wet, 225 - 150 * a, rgba[..., 1])
    rgba[..., 2] = np.where(wet, 255, rgba[..., 2])
    rgba[..., 3] = np.where(wet, (70 + 175 * a), rgba[..., 3]).astype(np.uint8)
    return _data_url(rgba)


def _map_png(values, colour_for):
    h, w = values.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    for code, col in colour_for.items():
        rgba[values == code] = col
    return _data_url(rgba)


# --------------------------------------------------------------------------
# 4. Full analysis (pure function of observations -> unit-testable without network)
# --------------------------------------------------------------------------

def analyse(obs: list, bbox: tuple, horizon_days: int, include_frames: bool = True) -> dict:
    if len(obs) < 8:
        raise HTTPException(422, f"Only {len(obs)} dates with NASA water products were found; at least 8 are needed "
                                 "to learn patterns. Try a longer look-back period.")
    h, w = obs[0]["valid"].shape
    Pn = h * w
    wst, s_, e_, n_ = bbox
    kmlat = (n_ - s_) * 111.32
    kmlon = (e_ - wst) * 111.32 * math.cos(math.radians((n_ + s_) / 2))
    px_km2 = kmlat * kmlon / Pn

    cover_all = np.array([o["valid"].mean() for o in obs])
    obs = [o for o, c in zip(obs, cover_all) if c >= MIN_FRAME_COVER]
    if len(obs) < 8:
        raise HTTPException(422, "Too few dates cover enough of the site (partial swaths only).")
    days = np.array([o["date"].toordinal() - EPOCH for o in obs], float)
    V = np.stack([o["valid"].ravel() for o in obs]).astype(np.float32)
    Y = np.stack([o["water"].ravel() for o in obs]).astype(np.float32)
    nobs = V.sum(0)
    domain = nobs >= MIN_PIXEL_OBS
    if domain.sum() < 50:
        raise HTTPException(422, "Too few pixels have enough repeated observations to model.")
    nd = int(domain.sum())
    span = float(days[-1] - days[0])

    notes = []
    if span < 300:
        notes.append("Less than ~1 year of data: annual seasonality cannot be estimated, so the forecast falls back "
                     "to each pixel's observed water frequency. Use a longer look-back period.")

    # ---- patterns (dedicated seasonal+trend fit on all data) -------------------------------
    dref = float(days.mean())
    Yd, Vd = Y[:, domain], V[:, domain]
    freq = ((Yd * Vd).sum(0) + 0.5) / (Vd.sum(0) + 1.0)
    pat_spec = SPECS["seasonal+trend"]
    if span >= 300:
        Bp, SEp = fit_logistic(design(days, pat_spec, dref), Yd, Vd, pat_spec["lam"], want_se=True)
        phase = np.linspace(0, 2 * np.pi, 25)[:-1]
        grid_days = phase / (2 * np.pi) * YEAR
        Xs = np.stack([np.ones(24), np.zeros(24), np.sin(phase), np.cos(phase)], 1)
        curve = _sigmoid(Xs @ Bp.T)                              # (24,P) seasonal cycle, trend at centre
        amp = curve.max(0) - curve.min(0)
        peak_doy = (np.argmax(curve, 0) * YEAR / 24).astype(float)
        slope = (curve * (1 - curve)).mean(0) * Bp[:, 1]         # d(prob)/year
        z = Bp[:, 1] / np.maximum(SEp[:, 1], 1e-6)
        sig_trend = (np.abs(z) > 2.5) & (np.abs(slope) > 0.05)
        site_curve = curve.mean(1) * 100
        site_peak = int(round(grid_days[int(np.argmax(site_curve))]))
    else:
        amp = np.zeros(nd); peak_doy = np.zeros(nd); slope = np.zeros(nd); sig_trend = np.zeros(nd, bool)
        site_curve = None; site_peak = None

    cls = np.zeros(nd, np.uint8)            # 1 permanent, 2 seasonal, 3 intermittent, 4 dry
    cls[freq <= 0.05] = 4
    cls[(freq > 0.05)] = 3
    cls[(freq > 0.05) & (amp >= 0.30)] = 2
    cls[freq >= 0.90] = 1
    full_cls = np.zeros(Pn, np.uint8); full_cls[domain] = cls
    full_freq = np.zeros(Pn); full_freq[domain] = freq
    trend_code = np.zeros(Pn, np.uint8)
    tc = np.zeros(nd, np.uint8); tc[sig_trend & (slope > 0)] = 1; tc[sig_trend & (slope < 0)] = 2
    trend_code[domain] = tc

    # ---- observed series over the modelled domain ------------------------------------------
    series = []
    for o, d in zip(obs, days):
        v = o["valid"].ravel()[domain]
        wv = o["water"].ravel()[domain]
        cov = float(v.mean())
        series.append({"date": o["date"].isoformat(), "cover": round(cov, 3),
                       "water_pct": round(float(wv[v].mean() * 100), 2) if v.any() else None,
                       "water_km2": round(float(wv.sum() * px_km2), 2)})
    good = [s for s in series if s["cover"] >= 0.5 and s["water_pct"] is not None]
    pulses = []
    if len(good) >= 8:
        vals = np.array([g["water_pct"] for g in good])
        med = np.median(vals); mad = np.median(np.abs(vals - med)) * 1.4826
        thr = med + 2 * max(mad, 0.5)
        pulses = [g["date"] for g in good if g["water_pct"] > thr]

    # ---- model selection by back-test, then refit on everything ----------------------------
    bt, best = backtest(days, Y, V, domain)
    if span < 300:
        best = "climatology"
    spec_name = best
    horizon = max(STEP_DAYS, min(int(horizon_days), 365))
    fdates = [obs[-1]["date"] + timedelta(days=STEP_DAYS * k) for k in range(1, int(math.ceil(horizon / STEP_DAYS)) + 1)]
    fdays = np.array([d.toordinal() - EPOCH for d in fdates], float)
    if spec_name in SPECS:
        spec = SPECS[spec_name]
        B = fit_logistic(design(days, spec, dref), Yd, Vd, spec["lam"])
        probs = _predict(B, spec, fdays, dref)                    # (M,nd)
    elif spec_name == "persistence":
        last = np.full(nd, np.nan)
        for t in range(Yd.shape[0]):
            last = np.where(Vd[t] > 0, Yd[t], last)
        probs = np.tile(np.clip(np.where(np.isnan(last), freq, last), 0.05, 0.95), (len(fdays), 1))
    else:
        spec_name = "climatology"
        probs = np.tile(freq, (len(fdays), 1))

    reliable, skill_c, skill_p = False, None, None
    rmse_frac = None
    if bt:
        m = bt["models"]
        if spec_name in m and "climatology" in m:
            skill_c = 1 - m[spec_name]["brier"] / max(m["climatology"]["brier"], 1e-9)
        if spec_name in m and "persistence" in m:
            skill_p = 1 - m[spec_name]["brier"] / max(m["persistence"]["brier"], 1e-9)
        rmse_frac = m.get(spec_name, {}).get("area_rmse_frac")
        reliable = bool(spec_name in SPECS and skill_c is not None and skill_c > 0.02 and (skill_p is None or skill_p > 0))
        if spec_name in ("climatology", "persistence"):
            notes.append(f"A simple '{spec_name}' baseline beat every learned model in the back-test, so it is used. "
                         "The surface here shows no stable seasonal/trend pattern that beats it.")
        elif not reliable:
            notes.append("The forecast model did not clearly beat the simple baselines in the back-test: treat the "
                         "forecast as low-confidence.")
    else:
        notes.append("Too few dates for a hold-out back-test: skill could not be measured, forecast is unverified.")
        reliable = False
    notes.append("Forecast uncertainty is calibrated on the back-test and does not widen with lead time; "
                 "extrapolating a trend far beyond the data is less certain than the band suggests.")

    forecast = []
    for k, d in enumerate(fdates):
        p = probs[k]
        exp_km2 = float(p.sum() * px_km2)
        sd_pix = float(np.sqrt((p * (1 - p)).sum()) * px_km2)
        sd_bt = float(rmse_frac * nd * px_km2) if rmse_frac is not None else 0.0
        sd = max(sd_pix, sd_bt)
        item = {"date": d.isoformat(), "lead_days": int((d - obs[-1]["date"]).days),
                "water_km2": round(exp_km2, 2), "low_km2": round(max(exp_km2 - 1.28 * sd, 0), 2),
                "high_km2": round(min(exp_km2 + 1.28 * sd, nd * px_km2), 2),
                "water_pct": round(exp_km2 / (nd * px_km2) * 100, 2)}
        if include_frames:
            item["png"] = _forecast_png(p, domain, (h, w))
        forecast.append(item)

    counts = {n: int((cls == c).sum()) for n, c in (("permanent", 1), ("seasonal", 2), ("intermittent", 3), ("dry", 4))}
    summary = {n: round(c * px_km2, 2) for n, c in counts.items()}
    summary.update({"gaining_km2": round(float((tc == 1).sum() * px_km2), 2),
                    "drying_km2": round(float((tc == 2).sum() * px_km2), 2),
                    "modelled_km2": round(nd * px_km2, 2),
                    "site_peak_day_of_year": site_peak})
    out = {
        "bbox": list(bbox), "grid": h, "px_km2": round(px_km2, 5),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": {"dates_used": len(obs), "first": obs[0]["date"].isoformat(), "last": obs[-1]["date"].isoformat(),
                 "span_days": int(span), "modelled_km2": summary["modelled_km2"]},
        "patterns": {"summary": summary, "timeseries": series, "pulses": pulses,
                     "seasonal_curve_pct": [round(float(x), 2) for x in site_curve] if site_curve is not None else None,
                     "legend": {"classes": [["#1648ff", "Permanent water (≥90% of dates)"],
                                            ["#ffb840", "Seasonal water (strong annual cycle)"],
                                            ["#8fd3ff", "Intermittent / flood-prone"], ["#2a3340", "Dry (≤5% of dates)"]],
                                "trend": [["#19e3ff", "Getting wetter"], ["#ff8c32", "Getting drier"]]}},
        "model": {"chosen": spec_name, "reliable": reliable, "skill_vs_climatology": skill_c,
                  "skill_vs_persistence": skill_p, "backtest": bt, "notes": notes,
                  "description": "Per-pixel logistic regression on annual harmonics (+ optional trend), fitted only on "
                                 "observed dates; compared against climatology and persistence on held-out real dates."},
    }
    if include_frames:
        out["patterns"]["maps"] = {
            "context": _context_png(obs, (h, w)),
            "classes": _map_png(full_cls.reshape(h, w), {1: (22, 72, 255, 235), 2: (255, 184, 64, 235),
                                                          3: (143, 211, 255, 215), 4: (42, 51, 64, 120)}),
            "trend": _map_png(trend_code.reshape(h, w), {1: (25, 227, 255, 235), 2: (255, 140, 50, 235)}),
        }
        out["frames"] = {"observed": [{"date": o["date"].isoformat(), "cover": s["cover"], "water_pct": s["water_pct"],
                                        "png": _observed_png(o)} for o, s in zip(obs, series)],
                         "forecast": forecast}
    else:
        out["frames"] = {"observed": [], "forecast": forecast}
    return out


# --------------------------------------------------------------------------
# 5. Background jobs (the GIBS fetch takes tens of seconds; the API polls)
# --------------------------------------------------------------------------

_JOBS: dict = {}
_RESULTS: dict = {}
_JLOCK = threading.Lock()
_RESULT_TTL = 3 * 3600
_MAX_JOBS = 24


def start_job(site: dict, lookback_days: int, horizon_days: int, grid: int, max_frames: int) -> dict:
    key = (site["id"], lookback_days, horizon_days, grid, max_frames)
    with _JLOCK:
        hit = _RESULTS.get(key)
        if hit and time.time() - hit[0] < _RESULT_TTL:
            jid = uuid.uuid4().hex[:12]
            _JOBS[jid] = {"state": "done", "created": time.time(), "result": hit[1], "progress": None}
            return {"job": jid, "state": "done"}
        for jid, j in _JOBS.items():     # reuse an identical running job
            if j.get("key") == key and j["state"] == "running":
                return {"job": jid, "state": "running"}
        while len(_JOBS) >= _MAX_JOBS:
            old = min(_JOBS, key=lambda k: _JOBS[k]["created"])
            _JOBS.pop(old, None)
        jid = uuid.uuid4().hex[:12]
        _JOBS[jid] = {"state": "running", "created": time.time(), "key": key,
                      "progress": {"stage": "Starting", "done": 0, "total": 0}}

    def progress(stage, done, total):
        with _JLOCK:
            if jid in _JOBS:
                _JOBS[jid]["progress"] = {"stage": stage, "done": done, "total": total}

    def run():
        try:
            bbox = tuple(float(v) for v in site["bbox"].split(","))
            obs, stats = fetch_observations(site["id"], bbox, grid, lookback_days, max_frames, progress)
            progress("Mining patterns and fitting the forecast", 0, 0)
            res = analyse(obs, bbox, horizon_days)
            res["site"] = site["id"]
            res["data"].update(stats)
            if stats["sar_dates_in_period"] > stats["dates_requested"]:
                res["model"]["notes"].append(
                    f"ASF lists {stats['sar_dates_in_period']} Sentinel-1 dates; {stats['dates_requested']} evenly spaced "
                    "dates were used to keep the request fast.")
            with _JLOCK:
                _RESULTS[key] = (time.time(), res)
                while len(_RESULTS) > 8:
                    _RESULTS.pop(next(iter(_RESULTS)), None)
                if jid in _JOBS:
                    _JOBS[jid].update(state="done", result=res, progress=None)
        except HTTPException as exc:
            with _JLOCK:
                if jid in _JOBS:
                    _JOBS[jid].update(state="error", error=str(exc.detail), status=exc.status_code, progress=None)
        except Exception as exc:                      # never leave a job hanging
            log.exception("forecast job failed")
            with _JLOCK:
                if jid in _JOBS:
                    _JOBS[jid].update(state="error", error=f"Forecast failed: {type(exc).__name__}", status=500, progress=None)

    threading.Thread(target=run, daemon=True).start()
    return {"job": jid, "state": "running"}


def job_status(jid: str) -> dict | None:
    with _JLOCK:
        j = _JOBS.get(jid)
        if not j:
            return None
        return {k: v for k, v in j.items() if k in ("state", "progress", "error", "status", "result")}
