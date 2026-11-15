"""
Dancing with the SARs — FastAPI backend.

Endpoints
---------
GET  /api/health    → status + configuration flags
GET  /api/sites     → the six study sites
GET  /api/timeline  → live SAR acquisitions (ASF DAAC) for one site
GET  /api/gibs      → unmodified NASA GIBS imagery for one site/day (PNG)
GET  /api/search    → raw GeoJSON search against ASF DAAC
POST /api/forecast  → start a surface-water pattern + forecast job (real NASA OPERA data)
GET  /api/forecast/{job} → poll a job (progress, then result)
POST /api/analyze   → SAR change detection on HDF5 pairs + overlay bounds
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from asf_client import search_nisar, site_timeline  # also loads .env
from gibs_client import fetch_gibs_png, LAYERS as GIBS_LAYERS
from forecast import start_job, job_status
from sar_processor import (
    compute_change_detection,
    generate_synthetic_sar_pair,
    load_h5_slice,
)

log = logging.getLogger("sars")
VERSION = "0.6.2"

app = FastAPI(
    title="Dancing with the SARs",
    version=VERSION,
    description="NASA Space Apps — SAR change-detection & flood-animation API",
)

# The bundled frontend is served from the same origin, so CORS is only needed for
# separately hosted frontends. Default: any origin, read-only methods, no credentials.
# Lock it down in production with e.g. CORS_ORIGINS="https://my-site.example".
_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,  # wildcard origins + credentials is invalid/insecure
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)  # timeline / analyze JSON compresses well


@app.middleware("http")
async def security_headers(request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    return resp


# ---------- Pydantic models ------------------------------------------------

ALLOWED_POLS = {"HHHH", "HVHV", "VVVV", "VHVH"}
SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"
DATA_ROOTS = [SAMPLE_DIR.resolve()] + (
    [Path(os.environ["SAR_DATA_DIR"]).resolve()] if os.environ.get("SAR_DATA_DIR") else []
)

SITES = [
    {"id": "prayagraj", "name": "Prayagraj (Sangam)", "country": "India", "lat": 25.43, "lon": 81.85,
     "bbox": "81.73,25.31,81.97,25.55", "kind": "Monsoon floodplain",
     "about": "Confluence of the Ganga and Yamuna. Seasonal monsoon flooding inundates the riverbanks and cropland.",
     "has_sample": False},
    {"id": "sacramento", "name": "Sacramento Valley", "country": "USA", "lat": 38.5, "lon": -121.6,
     "bbox": "-121.8,38.3,-121.4,38.7", "kind": "Flood basin / agriculture",
     "about": "Engineered flood-bypass system and rice country; winter atmospheric rivers flood the basins.",
     "has_sample": False},
    {"id": "dhaka", "name": "Padma-Meghna Delta", "country": "Bangladesh", "lat": 23.6, "lon": 90.5,
     "bbox": "90.3,23.4,90.7,23.8", "kind": "River delta",
     "about": "One of the world's most flood-prone deltas, where the Ganges, Brahmaputra and Meghna meet.",
     "has_sample": False},
    {"id": "mississippi", "name": "Mississippi Delta", "country": "USA", "lat": 29.3, "lon": -89.4,
     "bbox": "-89.6,29.1,-89.2,29.5", "kind": "Coastal wetland / subsidence",
     "about": "Sinking, storm-exposed wetlands at the mouth of the Mississippi; SAR tracks water and land loss.",
     "has_sample": False},
    {"id": "pantanal", "name": "Pantanal Wetland", "country": "Brazil", "lat": -17.8, "lon": -57.4,
     "bbox": "-57.6,-18.0,-57.2,-17.6", "kind": "Seasonal wetland",
     "about": "The largest tropical wetland on Earth, flooding and drying on an annual pulse that radar sees through cloud.",
     "has_sample": False},
    {"id": "valencia", "name": "Valencia / Albufera", "country": "Spain", "lat": 39.4, "lon": -0.4,
     "bbox": "-0.6,39.2,-0.2,39.6", "kind": "Flash-flood coast",
     "about": "Site of the catastrophic October 2024 DANA flash floods; rice paddies and a coastal lagoon.",
     "has_sample": False},
]


class AnalyzeRequest(BaseModel):
    pre_file: Optional[str] = None
    post_file: Optional[str] = None
    polarization: Optional[str] = "HHHH"
    threshold_db: Optional[float] = Field(default=3.0, ge=0.5, le=10.0)
    use_synthetic: Optional[bool] = False


def _safe_path(name: str) -> Path:
    """Resolve a user-supplied file name, allowing only the configured data roots
    (blocks reading arbitrary server files / path traversal)."""
    try:
        p = Path(name)
        p = p if p.is_absolute() else SAMPLE_DIR / p
        p = p.resolve()
    except (ValueError, OSError):
        raise HTTPException(status_code=400, detail="Invalid file name.")
    if not any(p == r or r in p.parents for r in DATA_ROOTS):
        raise HTTPException(status_code=400, detail="File must be inside the sample_data folder or SAR_DATA_DIR.")
    return p


def _iso(value: str, name: str) -> datetime:
    """Parse an ISO-8601 date/datetime query value or raise a clean 422."""
    try:
        v = value.strip().replace("Z", "+00:00")
        d = datetime.fromisoformat(v)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=422, detail=f"{name} must be an ISO-8601 date, e.g. 2026-01-31 or 2026-01-31T00:00:00Z")


# ---------- Endpoints -------------------------------------------------------

@app.get("/api/health")
async def health():
    token = os.environ.get("EARTHDATA_TOKEN", "").strip()
    has_token = bool(token and token != "your_earthdata_token_here")
    has_topo = bool(os.environ.get("OPENTOPOGRAPHY_API_KEY", "").strip())
    return {
        "status": "ok",
        "version": VERSION,
        "earthdata_token_configured": has_token,
        "opentopography_key_configured": has_topo,
    }


@app.get("/api/sites")
def sites():
    """Study sites the globe can fly to."""
    return SITES


@app.get("/api/timeline")
def timeline(
    site: str = Query(..., description="Site id from /api/sites"),
    start: Optional[str] = Query(default=None, description="ISO date; default = 12 months ago"),
    end: Optional[str] = Query(default=None, description="ISO date; default = now"),
):
    """Live, real SAR acquisitions (ASF DAAC) over one study site, sorted by time."""
    s = next((x for x in SITES if x["id"] == site), None)
    if s is None:
        raise HTTPException(status_code=404, detail="Unknown site id.")
    now = datetime.now(timezone.utc)
    d_end = _iso(end, "end") if end else now
    d_start = _iso(start, "start") if start else now - timedelta(days=365)
    if d_start >= d_end:
        raise HTTPException(status_code=422, detail="start must be before end.")
    if (d_end - d_start).days > 3 * 366:
        raise HTTPException(status_code=422, detail="Period too long (max 3 years).")
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    try:
        return site_timeline(s["id"], s["bbox"], d_start.astimezone(timezone.utc).strftime(fmt),
                             d_end.astimezone(timezone.utc).strftime(fmt))
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("timeline failed")
        raise HTTPException(status_code=502, detail=f"Timeline failed: {type(exc).__name__}")


@app.get("/api/gibs")
def gibs(
    layer: str = Query(..., description="rtc | wtr | true"),
    bbox: str = Query(..., description="west,south,east,north"),
    date: str = Query(..., description="YYYY-MM-DD"),
    size: int = Query(default=768, ge=64, le=1024),
):
    """Unmodified NASA GIBS imagery for one site/day (proxied so the browser can read pixels)."""
    try:
        png = fetch_gibs_png(layer, bbox, date, size)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("gibs failed")
        raise HTTPException(status_code=502, detail=f"GIBS failed: {type(exc).__name__}")
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=21600"})


@app.get("/api/search")
def search(
    bbox: str = Query(
        default="-122.5,37.5,-122.0,38.0",
        description="west,south,east,north (EPSG:4326)",
    ),
    start: str = Query(
        default="2024-01-01T00:00:00Z",
        description="ISO-8601 start date",
    ),
    end: str = Query(
        default="2024-12-31T23:59:59Z",
        description="ISO-8601 end date",
    ),
    max_results: int = Query(default=10, ge=1, le=100),
    platform: str = Query(
        default="NISAR,Sentinel-1",
        description="Platforms to query: NISAR,Sentinel-1, NISAR, Sentinel-1, or UAVSAR",
    ),
):
    """Proxy search to ASF DAAC and return GeoJSON."""
    geojson = search_nisar(
        bbox=bbox,
        start_date=start,
        end_date=end,
        max_results=max_results,
        platform=platform,
    )
    return geojson


@app.post("/api/forecast")
def forecast_start(
    site: str = Query(..., description="Site id from /api/sites"),
    lookback_days: int = Query(default=730, ge=180, le=1095, description="History to learn from (days)"),
    horizon_days: int = Query(default=96, ge=12, le=365, description="How far ahead to forecast (days)"),
    grid: int = Query(default=256, ge=96, le=384, description="Analysis grid (pixels per side)"),
    max_frames: int = Query(default=80, ge=12, le=120, description="Max Sentinel-1 dates to fetch"),
):
    """Start a background job: fetch real OPERA water/backscatter history from NASA GIBS, mine
    per-pixel patterns and forecast the future surface. Poll GET /api/forecast/{job}."""
    s = next((x for x in SITES if x["id"] == site), None)
    if s is None:
        raise HTTPException(status_code=404, detail="Unknown site id.")
    return start_job(s, lookback_days, horizon_days, grid, max_frames)


@app.get("/api/forecast/{job}")
def forecast_status(job: str):
    j = job_status(job)
    if j is None:
        raise HTTPException(status_code=404, detail="Unknown or expired forecast job.")
    if j["state"] == "error":
        raise HTTPException(status_code=j.get("status") or 500, detail=j.get("error") or "Forecast failed.")
    return j


@app.post("/api/analyze")
def analyze(body: AnalyzeRequest):
    """
    Run SAR change-detection.

    - If use_synthetic=True is explicitly requested: generates synthetic demo data.
    - If pre_file/post_file are provided (or "sample_pre"/"sample_post"): loads
      HDF5 observations.
    - Never silently substitutes synthetic data if real files fail or are missing.
    - The response always includes an explicit `mode` ("observed" | "synthetic").
    """
    sample_dir = SAMPLE_DIR
    sample_pre = sample_dir / "nisar_gcov_20240115_pre.h5"
    sample_post = sample_dir / "nisar_gcov_20240215_post.h5"

    pre_file = body.pre_file
    post_file = body.post_file

    # Map alias names to bundled sample dataset
    if pre_file == "sample_pre" or (pre_file is None and not body.use_synthetic and sample_pre.exists()):
        pre_file = str(sample_pre)
    if post_file == "sample_post" or (post_file is None and not body.use_synthetic and sample_post.exists()):
        post_file = str(sample_post)

    threshold = body.threshold_db if body.threshold_db is not None else 3.0
    pol = (body.polarization or "HHHH").upper()
    if pol not in ALLOWED_POLS:
        raise HTTPException(status_code=422, detail=f"polarization must be one of {sorted(ALLOWED_POLS)}")
    dataset_path = f"science/LSAR/GCOV/grids/frequencyA/{pol}"

    bounds = None
    provenance: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "polarization": pol,
        "threshold_db": threshold,
    }

    if body.use_synthetic:
        # Explicit synthetic mode requested
        pre_arr, post_arr = generate_synthetic_sar_pair()
        mode = "synthetic"
        bounds = {
            "west": -122.5,
            "south": 37.5,
            "east": -122.0,
            "north": 38.0,
        }
        provenance["shape"] = [512, 512]
        provenance["pre_file"] = "synthetic_field_A"
        provenance["post_file"] = "synthetic_field_B"
        provenance["note"] = (
            "Synthetic simulation requested. Results are purely demonstrative "
            "and MUST NOT be labeled as real satellite observations."
        )
    else:
        # Real observed data path
        if not pre_file or not post_file:
            raise HTTPException(
                status_code=400,
                detail="pre_file and post_file must be provided. For synthetic demo, pass use_synthetic=true.",
            )

        p_pre = _safe_path(pre_file)
        p_post = _safe_path(post_file)

        if not p_pre.is_file():
            raise HTTPException(
                status_code=404,
                detail="Pre-event HDF5 file not found. Will NOT silently substitute synthetic data.",
            )
        if not p_post.is_file():
            raise HTTPException(
                status_code=404,
                detail="Post-event HDF5 file not found. Will NOT silently substitute synthetic data.",
            )

        try:
            pre_arr, pre_meta = load_h5_slice(str(p_pre), dataset_path)
            post_arr, post_meta = load_h5_slice(str(p_post), dataset_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception:
            log.exception("HDF5 read failed")
            raise HTTPException(status_code=500, detail="Error reading HDF5 files (see server log).")

        # Shape compatibility check and crop to intersection
        if pre_arr.shape != post_arr.shape:
            min_rows = min(pre_arr.shape[0], post_arr.shape[0])
            min_cols = min(pre_arr.shape[1], post_arr.shape[1])
            pre_arr = pre_arr[:min_rows, :min_cols]
            post_arr = post_arr[:min_rows, :min_cols]

        bounds = pre_meta.get("bounds") or post_meta.get("bounds")
        # Bundled files are generated by create_sample_nisar_pair.py (NISAR-format, simulated
        # scene). They must never be labelled as real observations.
        is_sample = SAMPLE_DIR.resolve() in p_pre.parents and SAMPLE_DIR.resolve() in p_post.parents
        mode = "sample" if is_sample else "observed"
        if is_sample:
            provenance["note"] = "Simulated NISAR-format test pair; not a real satellite acquisition."
        provenance["pre_file"] = str(p_pre.name)
        provenance["post_file"] = str(p_post.name)
        provenance["shape"] = list(pre_arr.shape)
        provenance["crs"] = pre_meta.get("crs") or post_meta.get("crs")

    # --- change detection ---
    result = compute_change_detection(
        pre_arr,
        post_arr,
        nodata_value=0.0,
        mode=mode,
        threshold_db=threshold,
    )

    # Leaflet overlay bounds (SW, NE corners)
    leaflet_bounds = None
    if bounds:
        leaflet_bounds = [
            [bounds["south"], bounds["west"]],
            [bounds["north"], bounds["east"]],
        ]

    return {
        **result,
        "bounds": bounds,
        "leaflet_bounds": leaflet_bounds,
        "provenance": provenance,
    }


# Mount frontend static files so opening http://localhost:8000 loads the explorer directly
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.is_dir():
    from starlette.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

