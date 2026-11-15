"""
API endpoint validation tests for Dancing with the SARs (FastAPI).

Tests:
1. GET /api/health
2. GET /api/search (ASF DAAC proxy)
3. POST /api/analyze (Synthetic mode explicitly requested)
4. POST /api/analyze (Real observed HDF5 dataset)
5. POST /api/analyze (Error handling: missing files must NOT silently fall back to synthetic)
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
SAMPLE_PRE = SAMPLE_DIR / "nisar_gcov_20240115_pre.h5"
SAMPLE_POST = SAMPLE_DIR / "nisar_gcov_20240215_post.h5"


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_search_asf_daac(monkeypatch):
    """Verify ASF search proxy responds with GeoJSON (upstream mocked: no network in CI)."""
    import asf_client

    class _R:
        def raise_for_status(self): pass
        def json(self): return {"type": "FeatureCollection", "features": []}
    monkeypatch.setattr(asf_client.requests, "get", lambda *a, **k: _R())
    response = client.get(
        "/api/search",
        params={
            "bbox": "-122.5,37.5,-122.0,38.0",
            "start": "2024-01-01T00:00:00Z",
            "end": "2024-02-01T00:00:00Z",
            "max_results": 3,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("type") == "FeatureCollection"
    assert "features" in data
    # Features should have properties
    if len(data["features"]) > 0:
        feat = data["features"][0]
        assert "properties" in feat
        assert "geometry" in feat


def test_analyze_synthetic_explicit():
    """Verify explicitly requested synthetic mode returns mode='synthetic'."""
    response = client.post("/api/analyze", json={"use_synthetic": True})
    assert response.status_code == 200
    data = response.json()

    assert data["mode"] == "synthetic"
    assert "image_base64" in data
    assert data["image_base64"].startswith("data:image/png;base64,")
    assert "mean_db" in data
    assert "changed_area_pct" in data
    assert data["leaflet_bounds"] is not None
    assert "provenance" in data
    assert "synthetic" in data["provenance"]["note"].lower()


def test_analyze_real_sample_pair():
    """Verify observed analysis on reproducible NISAR GCOV sample pair."""
    if not (SAMPLE_PRE.exists() and SAMPLE_POST.exists()):
        pytest.skip("Sample files not present")

    response = client.post(
        "/api/analyze",
        json={
            "pre_file": str(SAMPLE_PRE),
            "post_file": str(SAMPLE_POST),
            "polarization": "HHHH",
            "threshold_db": 3.0,
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["mode"] == "sample"  # bundled pair is simulated, never "observed"
    assert "image_base64" in data
    assert data["mean_db"] != 0.0
    assert data["changed_area_pct"] > 0.0
    assert data["valid_pixel_count"] > 0
    assert data["total_pixel_count"] > data["valid_pixel_count"]  # Nodata masked
    assert data["threshold_db"] == 3.0

    # Leaflet bounds must be valid [[south, west], [north, east]]
    lb = data["leaflet_bounds"]
    assert len(lb) == 2
    south, west = lb[0]
    north, east = lb[1]
    assert south < north
    assert west < east

    # Provenance
    prov = data["provenance"]
    assert "nisar_gcov_20240115_pre.h5" in prov["pre_file"]
    assert "nisar_gcov_20240215_post.h5" in prov["post_file"]


def test_analyze_missing_file_rejected_without_silent_synthetic():
    """If a real file is requested but missing, fail with 404 instead of silent synthetic."""
    response = client.post(
        "/api/analyze",
        json={
            "pre_file": "nonexistent_pre_file.h5",
            "post_file": "nonexistent_post_file.h5",
            "use_synthetic": False,
        },
    )
    assert response.status_code == 404
    detail = response.json().get("detail", "")
    assert "not found" in detail.lower()


def test_png_aligns_with_raster_shape():
    """The overlay PNG must be exactly the raster (no colorbar/padding) so it fits the bounds."""
    import base64, io
    from PIL import Image
    r = client.post("/api/analyze", json={"use_synthetic": True}).json()
    im = Image.open(io.BytesIO(base64.b64decode(r["image_base64"].split(",")[1])))
    assert im.size[0] == im.size[1] == 512


def test_rejects_paths_outside_data_dir():
    r = client.post("/api/analyze", json={"pre_file": "/etc/passwd", "post_file": "/etc/hosts"})
    assert r.status_code == 400


def test_bad_polarization_and_bbox():
    assert client.post("/api/analyze", json={"use_synthetic": True, "polarization": "../x"}).status_code == 422
    assert client.get("/api/search", params={"bbox": "1,2,3"}).status_code == 422


def test_sites():
    assert any(s["id"] == "prayagraj" for s in client.get("/api/sites").json())


def test_sites_are_six():
    r = client.get("/api/sites")
    assert r.status_code == 200
    ids = [s["id"] for s in r.json()]
    assert len(ids) == 6 and len(set(ids)) == 6
    assert all(s["bbox"] and s["about"] for s in r.json())


def test_timeline_live_shape_and_order(monkeypatch):
    """Timeline merges ASF results, sorts by time and de-duplicates (upstream mocked)."""
    import asf_client
    asf_client._TL_CACHE.clear()

    def feat(name, t, plat):
        return {"type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
                "properties": {"sceneName": name, "startTime": t, "platform": plat,
                               "processingLevel": "GRD_HD", "flightDirection": "ASCENDING",
                               "pathNumber": 12, "polarization": "VV+VH"}}

    class _R:
        def __init__(self, plat): self.plat = plat
        def raise_for_status(self): pass
        def json(self):
            if self.plat == "NISAR":
                return {"features": [feat("N1", "2026-03-01T00:00:00Z", "NISAR")]}
            return {"features": [feat("S2", "2026-02-10T00:00:00Z", "Sentinel-1A"),
                                 feat("S1", "2026-01-01T00:00:00Z", "Sentinel-1A"),
                                 feat("S1", "2026-01-01T00:00:00Z", "Sentinel-1A")]}

    monkeypatch.setattr(asf_client.requests, "get",
                        lambda url, params=None, **k: _R(params["platform"]))
    r = client.get("/api/timeline", params={"site": "prayagraj"})
    assert r.status_code == 200
    j = r.json()
    assert [a["id"] for a in j["acquisitions"]] == ["S1", "S2", "N1"]
    assert j["source"].startswith("ASF DAAC")


def test_timeline_unknown_site():
    assert client.get("/api/timeline", params={"site": "nope"}).status_code == 404


def test_gibs_validation_and_proxy(monkeypatch):
    import gibs_client
    assert client.get("/api/gibs", params={"layer": "nope", "bbox": "0,0,1,1", "date": "2026-01-01"}).status_code == 422
    assert client.get("/api/gibs", params={"layer": "wtr", "bbox": "0,0,1,1", "date": "01-01-2026"}).status_code == 422
    assert client.get("/api/gibs", params={"layer": "wtr", "bbox": "0,0,9,9", "date": "2026-01-01"}).status_code == 422

    class _R:
        status_code = 200
        headers = {"content-type": "image/png"}
        content = b"\x89PNG-bytes"
    seen = {}
    def fake_get(url, params=None, **k):
        seen.update(params); return _R()
    monkeypatch.setattr(gibs_client.requests, "get", fake_get)
    gibs_client._CACHE.clear()
    r = client.get("/api/gibs", params={"layer": "wtr", "bbox": "81.73,25.31,81.97,25.55", "date": "2026-10-01"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert seen["LAYERS"] == "OPERA_L3_Dynamic_Surface_Water_Extent-Sentinel-1" and seen["TIME"] == "2026-10-01"


def test_gibs_missing_date_is_404(monkeypatch):
    import gibs_client
    class _R:
        status_code = 200
        headers = {"content-type": "text/xml"}
        content = b"<ServiceExceptionReport/>"
    monkeypatch.setattr(gibs_client.requests, "get", lambda *a, **k: _R())
    gibs_client._CACHE.clear()
    assert client.get("/api/gibs", params={"layer": "rtc", "bbox": "0,0,1,1", "date": "2026-01-02"}).status_code == 404


def test_health_reports_version():
    j = client.get("/api/health").json()
    assert j["status"] == "ok" and j["version"]


def test_timeline_rejects_bad_dates():
    assert client.get("/api/timeline", params={"site": "prayagraj", "start": "not-a-date"}).status_code == 422
    assert client.get("/api/timeline", params={"site": "prayagraj", "start": "2026-02-01", "end": "2026-01-01"}).status_code == 422
    assert client.get("/api/timeline", params={"site": "prayagraj", "start": "2020-01-01", "end": "2026-01-01"}).status_code == 422


def test_gibs_upstream_outage_is_502_not_404(monkeypatch):
    """A NASA outage must not be reported as 'no imagery for that day'."""
    import gibs_client

    class _R:
        status_code = 503
        headers = {"content-type": "text/html"}
        content = b""
    monkeypatch.setattr(gibs_client.requests, "get", lambda *a, **k: _R())
    gibs_client._CACHE.clear()
    assert client.get("/api/gibs", params={"layer": "wtr", "bbox": "0,0,1,1", "date": "2026-01-05"}).status_code == 502


def test_asf_invalid_token_falls_back_to_anonymous_search(monkeypatch):
    import asf_client
    import requests
    monkeypatch.setenv("EARTHDATA_TOKEN", "expired-token")
    sent = []

    class _R:
        def __init__(self, code): self.status_code = code
        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.exceptions.HTTPError("denied")
        def json(self): return {"type": "FeatureCollection", "features": []}

    def fake_get(url, params=None, headers=None, **k):
        sent.append(bool(headers))
        return _R(401 if headers else 200)
    monkeypatch.setattr(asf_client.requests, "get", fake_get)
    assert client.get("/api/search", params={"platform": "Sentinel-1"}).status_code == 200
    assert sent == [True, False]


def test_security_headers_present():
    r = client.get("/api/health")
    assert r.headers["x-content-type-options"] == "nosniff"
