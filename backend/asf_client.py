"""
ASF DAAC client for searching NISAR / Sentinel-1 SAR granules.
"""

import os
from pathlib import Path
import requests
from fastapi import HTTPException

def load_env() -> None:
    """Load backend/.env once (shared by every module)."""
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


load_env()

PLATFORMS = {"NISAR,Sentinel-1", "NISAR", "Sentinel-1", "UAVSAR"}

ASF_SEARCH_URL = "https://api.daac.asf.alaska.edu/services/search/param"


def get_earthdata_headers() -> dict:
    """Return Bearer auth header if EARTHDATA_TOKEN is set, else empty dict."""
    token = os.environ.get("EARTHDATA_TOKEN", "").strip()
    if token and token != "your_earthdata_token_here":
        return {"Authorization": f"Bearer {token}"}
    return {}


def search_nisar(
    bbox: str,
    start_date: str,
    end_date: str,
    max_results: int = 10,
    platform: str = "NISAR,Sentinel-1",
) -> dict:
    """
    Search ASF DAAC for SAR granules across NASA & partner missions.

    Parameters
    ----------
    bbox : str
        Bounding box as "west,south,east,north" (EPSG:4326).
    start_date : str
        ISO-8601 start date, e.g. "2024-01-01T00:00:00Z".
    end_date : str
        ISO-8601 end date.
    max_results : int
        Maximum number of results to return.
    platform : str
        Platform filter: "NISAR,Sentinel-1", "NISAR", "Sentinel-1", or "UAVSAR".

    Returns
    -------
    dict
        GeoJSON FeatureCollection of matching granules.

    Raises
    ------
    HTTPException(502)
        When the upstream ASF API call fails.
    """
    if platform not in PLATFORMS:
        raise HTTPException(status_code=422, detail=f"platform must be one of {sorted(PLATFORMS)}")
    try:
        w, s_, e, n = (float(x) for x in bbox.split(","))
        assert -180 <= w < e <= 180 and -90 <= s_ < n <= 90
    except Exception:
        raise HTTPException(status_code=422, detail="bbox must be 'west,south,east,north' in degrees (west < east, south < north).")
    params = {
        "platform": platform,
        "bbox": bbox,
        "start": start_date,
        "end": end_date,
        "maxResults": max_results,
        "output": "geojson",
    }

    # Set appropriate processing levels per platform
    if "NISAR" in platform and "Sentinel-1" in platform:
        params["processingLevel"] = "GCOV,GRD_HD,GSLC,RSLC"
    elif platform == "NISAR":
        params["processingLevel"] = "GCOV,GSLC,RSLC"
    elif platform == "Sentinel-1":
        params["processingLevel"] = "GRD_HD"

    headers = get_earthdata_headers()

    last_exc = None
    for attempt in range(2):  # one retry for timeouts / transient 5xx
        try:
            resp = requests.get(ASF_SEARCH_URL, params=params, headers=headers, timeout=30)
            if headers and getattr(resp, "status_code", 200) in (401, 403):
                # Search is public: a stale/invalid token must not break it. Retry anonymously.
                headers = {}
                resp = requests.get(ASF_SEARCH_URL, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            transient = isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)) \
                or (status is not None and status >= 500)
            if not transient:
                break
        except ValueError as exc:  # invalid JSON from upstream
            raise HTTPException(status_code=502, detail="ASF DAAC returned an unreadable response.") from exc

    if isinstance(last_exc, requests.exceptions.Timeout):
        raise HTTPException(status_code=502, detail="ASF DAAC search timed out after 30 s.")
    detail = f"ASF DAAC search failed: {type(last_exc).__name__}"
    if getattr(last_exc, "response", None) is not None:
        detail += f" (HTTP {last_exc.response.status_code}) — {last_exc.response.text[:200]}"
    raise HTTPException(status_code=502, detail=detail)


# ---------------------------------------------------------------------------
# Live per-site acquisition timeline (real ASF metadata, no synthetic data)
# ---------------------------------------------------------------------------
import time
from datetime import datetime, timezone

_TL_CACHE: dict = {}
_TL_TTL = 15 * 60  # seconds
_TL_MAX = 64       # bounded so a long-running server cannot grow without limit


def _footprint(geom) -> list:
    """Outer ring of a GeoJSON polygon as [[lon, lat], ...] (or [])."""
    try:
        if geom["type"] == "Polygon":
            return [[round(x, 4), round(y, 4)] for x, y in geom["coordinates"][0]]
        if geom["type"] == "MultiPolygon":
            return [[round(x, 4), round(y, 4)] for x, y in geom["coordinates"][0][0]]
    except Exception:
        pass
    return []


def _mb(v):
    try:
        return round(float(v) / 1e6, 1)
    except (TypeError, ValueError):
        return None


def _slim(feat: dict) -> dict:
    p = feat.get("properties", {}) or {}
    return {
        "id": p.get("sceneName") or p.get("fileID") or p.get("fileName"),
        "time": p.get("startTime"),
        "platform": p.get("platform") or p.get("sensor"),
        "level": p.get("processingLevel"),
        "mode": p.get("beamModeType") or p.get("beamMode"),
        "pol": p.get("polarization"),
        "direction": p.get("flightDirection"),
        "path": p.get("pathNumber"),
        "frame": p.get("frameNumber"),
        "orbit": p.get("orbit"),
        "size_mb": _mb(p.get("bytes")),
        "url": p.get("url"),
        "footprint": _footprint(feat.get("geometry")),
    }


def site_timeline(site_id: str, bbox: str, start: str, end: str, max_results: int = 200) -> dict:
    """
    Real SAR acquisitions over `bbox` between `start` and `end`, sorted by time.
    Queries ASF DAAC live for Sentinel-1 and NISAR; results are cached for 15 min.
    Platforms that fail are reported in `errors`; nothing is ever fabricated.
    """
    cache_key = (site_id, bbox, start[:10], end[:10])
    hit = _TL_CACHE.get(cache_key)
    if hit and time.time() - hit[0] < _TL_TTL:
        return hit[1]

    items, errors = [], {}
    for plat in ("Sentinel-1", "NISAR"):
        try:
            gj = search_nisar(bbox=bbox, start_date=start, end_date=end,
                              max_results=max_results, platform=plat)
            feats = gj.get("features", []) if isinstance(gj, dict) else []
            for f in feats:
                try:
                    items.append(_slim(f))
                except Exception:
                    continue  # skip a malformed record, keep the rest
        except HTTPException as exc:
            errors[plat] = str(exc.detail)
        except Exception as exc:  # JSON decode errors, unexpected upstream shapes, etc.
            errors[plat] = f"{type(exc).__name__}: {exc}"
    if not items and len(errors) == 2:
        raise HTTPException(status_code=502, detail="ASF DAAC request failed: " + " | ".join(f"{k}: {v}" for k, v in errors.items()))

    seen, out = set(), []
    for it in sorted((i for i in items if i["time"]), key=lambda i: i["time"]):
        key_ = it["id"] or (it["time"], it["platform"], it["path"], it["frame"])
        if key_ in seen:
            continue
        seen.add(key_); out.append(it)

    result = {
        "site": site_id, "bbox": bbox, "start": start, "end": end,
        "source": "ASF DAAC (live)", "fetched_at": datetime.now(timezone.utc).isoformat(),
        "count": len(out), "errors": errors, "acquisitions": out,
    }
    while len(_TL_CACHE) >= _TL_MAX:
        _TL_CACHE.pop(next(iter(_TL_CACHE)), None)
    _TL_CACHE[cache_key] = (time.time(), result)
    return result
