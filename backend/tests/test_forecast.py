"""
Tests for the pattern-mining + forecast feature (forecast.py, /api/forecast).

The numeric tests use a synthetic stack ONLY to verify the maths (the model must recover a
known seasonal signal); the API test mocks GIBS/ASF so no network is needed. The shipped
feature itself never uses synthetic data.
"""
import io
import time
from datetime import date, timedelta

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

import forecast as F
from main import app

client = TestClient(app)
BBOX = (81.73, 25.31, 81.97, 25.55)


def _stack(n=75, size=48, seed=1):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    perm, seas = xx < size // 4, (xx >= size // 4) & (xx < size * 5 // 8)
    obs, d0 = [], date(2023, 11, 1)
    for k in range(n):
        d = d0 + timedelta(days=12 * k)
        s = 0.5 + 0.5 * np.sin(2 * np.pi * d.toordinal() / 365.25 - 1.0)
        water = perm | (seas & (rng.random((size, size)) < 0.9 * s ** 2))
        swath = np.zeros((size, size), bool)
        swath[:, (k * 7) % 12:(k * 7) % 12 + 34] = True      # moving swath: part of the site unseen each date
        obs.append({"date": d, "valid": swath, "water": water & swath, "has_wtr": True,
                    "lum": np.where(swath, 100 + 30 * perm, np.nan).astype(np.float32)})
    return obs, perm, seas


def test_classify_dswx_colours():
    img = np.zeros((1, 4, 4), np.uint8)
    img[0, 0] = (20, 40, 255, 255)    # open water
    img[0, 1] = (30, 200, 60, 255)    # inundated vegetation
    img[0, 2] = (170, 175, 180, 255)  # masked
    img[0, 3] = (0, 0, 0, 0)          # transparent
    water, has, masked = F.classify_wtr(img)
    assert water[0].tolist() == [True, True, False, False]
    assert has[0].tolist() == [True, True, True, False]
    assert masked[0].tolist() == [False, False, True, False]


def test_unobserved_pixels_are_never_dry():
    fp = np.zeros((8, 8), bool); fp[:, :4] = True
    wtr = np.zeros((8, 8, 4), np.uint8); wtr[:, :2] = (20, 40, 255, 255)
    ob = F.make_observation("2025-01-01", None, wtr, fp)
    assert ob["valid"][:, 4:].sum() == 0 and not ob["water"][:, 4:].any()
    assert ob["water"][:, :2].all()
    assert F.make_observation("2025-01-01", None, None, fp) is None


def test_radar_without_water_product_is_not_an_observation():
    fp = np.ones((8, 8), bool)
    rtc = np.zeros((8, 8, 4), np.uint8); rtc[..., 3] = 255
    assert F.make_observation("2025-01-01", rtc, None, fp) is None      # must not turn into "all dry"


def test_missing_radar_layer_keeps_land_valid():
    fp = np.ones((8, 8), bool)
    wtr = np.zeros((8, 8, 4), np.uint8); wtr[:, :2] = (20, 40, 255, 255)
    ob = F.make_observation("2025-01-01", None, wtr, fp)
    assert ob["valid"].all() and ob["water"][:, :2].all() and not ob["water"][:, 2:].any()


def test_model_recovers_seasonality_and_beats_baselines():
    obs, perm, seas = _stack()
    r = F.analyse(obs, BBOX, horizon_days=120)
    m = r["model"]
    assert m["chosen"] in F.SPECS and m["reliable"] is True
    assert m["skill_vs_climatology"] > 0.1
    bt = m["backtest"]["models"]
    assert bt[m["chosen"]]["brier"] < bt["climatology"]["brier"]
    s = r["patterns"]["summary"]
    px = r["px_km2"]
    assert s["permanent"] == pytest.approx(perm.sum() * px, rel=0.15)
    assert s["seasonal"] > 0
    f = r["frames"]["forecast"]
    assert len(f) == 10 and f[0]["lead_days"] == 12
    assert all(x["low_km2"] <= x["water_km2"] <= x["high_km2"] for x in f)
    assert f[0]["png"].startswith("data:image/png;base64,")
    # forecast follows the seasonal cycle instead of staying flat
    assert max(x["water_pct"] for x in f) - min(x["water_pct"] for x in f) > 3


def test_peak_day_of_year_is_a_calendar_day():
    rng = np.random.default_rng(5)
    size, obs = 32, []
    for k in range(90):
        d = date(2023, 11, 1) + timedelta(days=12 * k)
        doy = d.timetuple().tm_yday
        p = 0.5 + 0.45 * np.cos(2 * np.pi * (doy - 200) / 365.25)      # wettest ~19 July
        obs.append({"date": d, "valid": np.ones((size, size), bool), "lum": None, "has_wtr": True,
                    "water": rng.random((size, size)) < p})
    peak = F.analyse(obs, BBOX, horizon_days=24)["patterns"]["summary"]["site_peak_day_of_year"]
    assert abs(peak - 200) <= 12


def test_short_record_falls_back_and_says_so():
    obs, _, _ = _stack(n=16)               # 16 x 12 d = 180 days < 1 year
    r = F.analyse(obs, BBOX, horizon_days=48)
    assert r["model"]["chosen"] == "climatology" and r["model"]["reliable"] is False
    assert any("1 year" in n for n in r["model"]["notes"])


def test_too_little_data_is_refused():
    obs, _, _ = _stack(n=5)
    with pytest.raises(HTTPException) as e:
        F.analyse(obs, BBOX, horizon_days=48)
    assert e.value.status_code == 422


def test_trend_detection():
    rng = np.random.default_rng(3)
    size, obs = 40, []
    for k in range(80):
        d = date(2023, 11, 1) + timedelta(days=12 * k)
        p = 0.1 + 0.8 * k / 79                       # steadily wetter in the right half
        w = np.zeros((size, size), bool); w[:, 20:] = rng.random((size, 20)) < p
        v = np.ones((size, size), bool)
        obs.append({"date": d, "valid": v, "water": w, "lum": None, "has_wtr": True})
    r = F.analyse(obs, BBOX, horizon_days=24)
    sm = r["patterns"]["summary"]
    assert sm["gaining_km2"] > 0.25 * sm["modelled_km2"]      # the wetter half is detected
    assert sm["drying_km2"] == 0


# ---------------- API (GIBS + ASF mocked) ----------------

def _png(rgba):
    b = io.BytesIO(); Image.fromarray(rgba, "RGBA").save(b, "PNG"); return b.getvalue()


@pytest.fixture
def mocked(monkeypatch):
    def fake_gibs(layer, bbox, day, grid):
        size = grid
        yy, xx = np.mgrid[0:size, 0:size]
        xx = xx * 64 // size                      # same scene at any grid size
        d = date.fromisoformat(day)
        rng = np.random.default_rng(d.toordinal())
        if layer == "rtc":
            a = np.zeros((grid, grid, 4), np.uint8); a[..., :3] = 120; a[..., 3] = 255
            return _png(a)
        s = 0.5 + 0.5 * np.sin(2 * np.pi * d.toordinal() / 365.25 - 1.0)
        water = (xx < 16) | ((xx >= 16) & (xx < 40) & (rng.random((size, size)) < 0.9 * s ** 2))
        a = np.zeros((grid, grid, 4), np.uint8); a[water] = (20, 40, 255, 255)
        return _png(a)

    def fake_tl(site_id, bbox, start, end, max_results=200):
        s, e = date.fromisoformat(start[:10]), date.fromisoformat(end[:10])
        fp = [[81.6, 25.2], [82.1, 25.2], [82.1, 25.7], [81.6, 25.7], [81.6, 25.2]]
        acq, d = [], s
        while d <= e:
            acq.append({"id": f"S1_{d}", "time": f"{d}T00:00:00Z", "platform": "Sentinel-1A", "footprint": fp})
            d += timedelta(days=12)
        return {"acquisitions": acq, "errors": {}}

    monkeypatch.setattr(F, "fetch_gibs_png", fake_gibs)
    monkeypatch.setattr(F, "site_timeline", fake_tl)
    F._RESULTS.clear(); F._JOBS.clear()


def _wait(job, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(f"/api/forecast/{job}")
        if r.status_code != 200 or r.json()["state"] != "running":
            return r
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_forecast_api_end_to_end(mocked):
    r = client.post("/api/forecast", params={"site": "prayagraj", "lookback_days": 900, "horizon_days": 60,
                                              "grid": 96, "max_frames": 70})
    assert r.status_code == 200
    out = _wait(r.json()["job"])
    assert out.status_code == 200, out.text
    j = out.json()
    assert j["state"] == "done"
    res = j["result"]
    assert res["site"] == "prayagraj" and res["data"]["dates_used"] >= 40
    assert res["frames"]["observed"] and res["frames"]["forecast"]
    assert res["patterns"]["maps"]["context"].startswith("data:image/png")


def test_forecast_api_validation(mocked):
    assert client.post("/api/forecast", params={"site": "nope"}).status_code == 404
    assert client.post("/api/forecast", params={"site": "prayagraj", "lookback_days": 10}).status_code == 422
    assert client.post("/api/forecast", params={"site": "prayagraj", "grid": 5000}).status_code == 422
    assert client.get("/api/forecast/doesnotexist").status_code == 404


def test_forecast_api_reports_nasa_outage(monkeypatch):
    def boom(*a, **k):
        raise HTTPException(502, "down")
    monkeypatch.setattr(F, "fetch_gibs_png", boom)
    monkeypatch.setattr(F, "site_timeline", lambda *a, **k: {"acquisitions": [
        {"id": "x", "time": "2025-01-01T00:00:00Z", "platform": "Sentinel-1A", "footprint": []}], "errors": {}})
    F._RESULTS.clear(); F._JOBS.clear()
    r = client.post("/api/forecast", params={"site": "prayagraj"})
    out = _wait(r.json()["job"])
    assert out.status_code == 502
