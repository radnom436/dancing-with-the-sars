"""
NASA GIBS (Global Imagery Browse Services) proxy.

Fetches *unmodified* rendered NASA imagery for one site bbox and one calendar
day through GIBS' public WMS endpoint (no API key needed), so the browser can
read pixels without cross-origin restrictions.

Layers (all NASA / NASA-hosted):
  rtc   OPERA L2 Radiometric Terrain Corrected SAR backscatter (Sentinel-1)
  wtr   OPERA L3 Dynamic Surface Water Extent from Sentinel-1 (DSWx-S1)
  true  MODIS Terra corrected-reflectance true colour (optical context)
"""
from __future__ import annotations

import re
import threading
import time
from datetime import date as _date

import requests
from fastapi import HTTPException

GIBS_WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"

LAYERS = {
    "rtc": "OPERA_L2_Radiometric_Terrain_Corrected_SAR_Sentinel-1",
    "wtr": "OPERA_L3_Dynamic_Surface_Water_Extent-Sentinel-1",
    "true": "MODIS_Terra_CorrectedReflectance_TrueColor",
}

_CACHE: dict = {}
_LOCK = threading.Lock()  # the route runs in a thread pool; guard cache mutation
_TTL = 6 * 3600
_MAX = 400


def fetch_gibs_png(layer: str, bbox: str, day: str, size: int = 768) -> bytes:
    if layer not in LAYERS:
        raise HTTPException(422, f"layer must be one of {sorted(LAYERS)}")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day or ""):
        raise HTTPException(422, "date must be YYYY-MM-DD")
    try:
        _date.fromisoformat(day)
        w, s, e, n = (float(x) for x in bbox.split(","))
        assert -180 <= w < e <= 180 and -90 <= s < n <= 90 and (e - w) <= 2 and (n - s) <= 2
    except Exception:
        raise HTTPException(422, "bbox must be 'west,south,east,north' (max 2 degrees span).")
    if not 64 <= size <= 1024:
        raise HTTPException(422, "size must be 64..1024")

    key = (layer, round(w, 4), round(s, 4), round(e, 4), round(n, 4), day, size)
    with _LOCK:
        hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]

    params = {
        "SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap",
        "LAYERS": LAYERS[layer], "STYLES": "", "SRS": "EPSG:4326",
        "BBOX": f"{w},{s},{e},{n}", "WIDTH": size, "HEIGHT": size,
        "FORMAT": "image/png", "TRANSPARENT": "TRUE", "TIME": day,
    }
    r = None
    for attempt in range(2):  # one retry for transient network errors / 5xx
        try:
            r = requests.get(GIBS_WMS, params=params, timeout=40)
        except requests.exceptions.RequestException as exc:
            if attempt:
                raise HTTPException(502, f"GIBS request failed: {type(exc).__name__}")
            continue
        if r.status_code >= 500 and not attempt:
            continue
        break
    ctype = r.headers.get("content-type", "")
    if r.status_code >= 500:
        raise HTTPException(502, f"NASA GIBS is temporarily unavailable (HTTP {r.status_code}).")
    if r.status_code != 200 or not ctype.startswith("image/"):
        raise HTTPException(404, f"No GIBS imagery for {LAYERS[layer]} on {day} (HTTP {r.status_code}).")

    with _LOCK:
        while len(_CACHE) >= _MAX:
            _CACHE.pop(next(iter(_CACHE)), None)
        _CACHE[key] = (time.time(), r.content)
    return r.content
