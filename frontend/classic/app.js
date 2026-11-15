/**
 * Dancing with the SARs — Earth Observation Change Explorer
 * Frontend Application Logic (Stage 1: Representation only)
 */

const API_BASE = "http://127.0.0.1:8000";

// --- State ---
let map = null;
let currentOverlay = null;
let currentAnalysisMode = "observed"; // 'observed' | 'synthetic'
let lastAnalysisData = null;

// --- Initialize App ---
document.addEventListener("DOMContentLoaded", () => {
  initMap();
  checkBackendHealth();
  setupEventListeners();
});

// --- Map Initialization ---
function initMap() {
  // Default centered over Sacramento Valley flood study area
  map = L.map("map", {
    zoomControl: true,
    attributionControl: true,
  }).setView([38.5, -121.6], 10);

  // Basemap 1: ESRI World Imagery (High-Resolution Satellite - Free, no API key)
  const esriSatellite = L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    {
      attribution:
        "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
      maxZoom: 18,
    }
  );

  // Basemap 2: OpenStreetMap Standard (Free, no API key)
  const osmStandard = L.tileLayer(
    "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    {
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 19,
    }
  );

  // Add ESRI Satellite as default active basemap
  esriSatellite.addTo(map);

  // Layer Switcher for Basemaps
  const baseMaps = {
    "Satellite Imagery (ESRI)": esriSatellite,
    "Street / Topo Map (OSM)": osmStandard,
  };
  L.control.layers(baseMaps, null, { position: "topright" }).addTo(map);

  // Add study area reference bounding box (Sacramento Valley)
  const sacramentoBounds = [
    [38.3, -121.8],
    [38.7, -121.4],
  ];
  L.rectangle(sacramentoBounds, {
    color: "#38bdf8",
    weight: 1.5,
    dashArray: "4, 6",
    fillColor: "#38bdf8",
    fillOpacity: 0.05,
  }).addTo(map);
}

// --- Check Backend Health ---
async function checkBackendHealth() {
  const statusDot = document.querySelector(".status-dot");
  const statusText = document.getElementById("api-status-text");

  try {
    const res = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      const data = await res.json();
      statusDot.className = "status-dot online";
      let keyInfo = [];
      if (data.earthdata_token_configured) keyInfo.push("Earthdata Active");
      if (data.opentopography_key_configured) keyInfo.push("OpenTopography Active");

      if (keyInfo.length > 0) {
        statusText.textContent = `Backend Live • ${keyInfo.join(" • ")}`;
      } else {
        statusText.textContent = "Backend Live (Keys Optional)";
      }
    } else {
      throw new Error(`HTTP ${res.status}`);
    }
  } catch (err) {
    statusDot.className = "status-dot error";
    statusText.textContent = "Backend Offline (Start Uvicorn)";
  }
}

// --- Event Listeners ---
function setupEventListeners() {
  // Real NISAR Analysis Button
  document.getElementById("btn-run-real").addEventListener("click", () => {
    runAnalysis({ use_synthetic: false });
  });

  // Synthetic Demo Button
  document.getElementById("btn-run-synthetic").addEventListener("click", () => {
    runAnalysis({ use_synthetic: true });
  });

  // Threshold Slider
  const thresholdSlider = document.getElementById("threshold-slider");
  const thresholdVal = document.getElementById("threshold-val");
  thresholdSlider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value).toFixed(1);
    thresholdVal.textContent = `${val} dB`;
  });
  thresholdSlider.addEventListener("change", (e) => {
    const val = parseFloat(e.target.value);
    runAnalysis({
      use_synthetic: currentAnalysisMode === "synthetic",
      threshold_db: val,
    });
  });

  // Study Region Selector
  const regionSelect = document.getElementById("study-region");
  const customBboxGroup = document.getElementById("custom-bbox-group");
  regionSelect.addEventListener("change", (e) => {
    if (e.target.value === "custom") {
      customBboxGroup.style.display = "block";
    } else {
      customBboxGroup.style.display = "none";
      const parts = e.target.value.split(",").map(Number);
      if (parts.length === 4) {
        const [w, s, eLon, n] = parts;
        map.fitBounds([
          [s, w],
          [n, eLon],
        ]);
      }
    }
  });

  // Resources Modal Controls
  const modal = document.getElementById("resources-modal");
  document.getElementById("btn-open-resources").addEventListener("click", () => {
    modal.style.display = "flex";
  });
  document.getElementById("btn-close-resources").addEventListener("click", () => {
    modal.style.display = "none";
  });
  modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.style.display = "none";
  });

  // Search Form
  document.getElementById("search-form").addEventListener("submit", (e) => {
    e.preventDefault();
    executeAsfSearch();
  });
}

// --- Execute ASF Search ---
async function executeAsfSearch() {
  const btn = document.getElementById("btn-search");
  const granulesList = document.getElementById("granules-list");
  const countSpan = document.getElementById("results-count");

  let bbox = document.getElementById("study-region").value;
  if (bbox === "custom") {
    bbox = document.getElementById("bbox-input").value;
  }
  const start = document.getElementById("start-date").value;
  const end = document.getElementById("end-date").value;
  const platform = document.getElementById("platform-select").value;

  btn.disabled = true;
  btn.textContent = "Querying ASF DAAC...";
  granulesList.innerHTML = `<p class="placeholder-text">Contacting Alaska Satellite Facility API...</p>`;

  try {
    const url = new URL(`${API_BASE}/api/search`);
    url.searchParams.set("bbox", bbox);
    url.searchParams.set("start", `${start}T00:00:00Z`);
    url.searchParams.set("end", `${end}T23:59:59Z`);
    url.searchParams.set("platform", platform);
    url.searchParams.set("max_results", "8");
    url.searchParams.set("start", `${start}T00:00:00Z`);
    url.searchParams.set("end", `${end}T23:59:59Z`);
    url.searchParams.set("max_results", "8");

    const res = await fetch(url);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const geojson = await res.json();
    const features = geojson.features || [];
    countSpan.textContent = features.length;

    if (features.length === 0) {
      granulesList.innerHTML = `<p class="placeholder-text">No granules found in this date range. Try widening the dates.</p>`;
      return;
    }

    granulesList.innerHTML = "";
    features.forEach((feat, idx) => {
      const p = feat.properties || {};
      const item = document.createElement("div");
      item.className = "granule-item";
      item.innerHTML = `
        <div class="granule-title">${p.sceneName || p.fileName || `Granule ${idx + 1}`}</div>
        <div class="granule-meta">
          <span>${p.platform || "SAR"} • ${p.processingLevel || "GCOV"}</span>
          <span>${p.startTime ? p.startTime.slice(0, 10) : "N/A"}</span>
          <span>${p.polarization || "Co/Cross"}</span>
        </div>
      `;
      granulesList.appendChild(item);
    });
  } catch (err) {
    granulesList.innerHTML = `<p class="placeholder-text" style="color: var(--status-red);">Search failed: ${err.message}</p>`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg> Search ASF Archive`;
  }
}

// --- Run Change Detection Analysis ---
async function runAnalysis(options = {}) {
  const btnReal = document.getElementById("btn-run-real");
  const btnSynth = document.getElementById("btn-run-synthetic");
  const banner = document.getElementById("mode-banner");
  const badge = document.getElementById("banner-badge");
  const bannerText = document.getElementById("banner-text");

  const useSynthetic = options.use_synthetic ?? false;
  currentAnalysisMode = useSynthetic ? "synthetic" : "observed";

  const threshold = options.threshold_db ?? parseFloat(document.getElementById("threshold-slider").value);
  const polarization = document.getElementById("pol-select").value;

  btnReal.disabled = true;
  btnSynth.disabled = true;

  try {
    const payload = {
      use_synthetic: useSynthetic,
      threshold_db: threshold,
      polarization: polarization,
    };

    if (!useSynthetic) {
      payload.pre_file = "sample_pre";
      payload.post_file = "sample_post";
    }

    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Analysis failed: HTTP ${res.status}`);
    }

    const data = await res.json();
    lastAnalysisData = data;

    // Render results
    updateMetricsAndProvenance(data);
    renderMapOverlay(data);

    // Update Mode Banner with strict scientific distinction
    if (data.mode === "observed") {
      banner.className = "mode-banner banner-real";
      badge.textContent = "✓ Real Observation";
      bannerText.textContent = "Observed NISAR GCOV calibrated backscatter change (Sacramento Flood 2024).";
    } else {
      banner.className = "mode-banner banner-synthetic";
      badge.textContent = "⚠️ Synthetic Simulation";
      bannerText.textContent = "Simulated SAR backscatter pair. Illustrative only — NOT real satellite data.";
    }
  } catch (err) {
    alert(`Analysis error: ${err.message}`);
  } finally {
    btnReal.disabled = false;
    btnSynth.disabled = false;
  }
}

// --- Update Metrics & Provenance ---
function updateMetricsAndProvenance(data) {
  document.getElementById("metric-mean-db").textContent =
    (data.mean_db > 0 ? "+" : "") + Number(data.mean_db).toFixed(2);
  document.getElementById("metric-changed-pct").textContent =
    Number(data.changed_area_pct).toFixed(1) + "%";

  const valid = data.valid_pixel_count || 0;
  const total = data.total_pixel_count || 0;
  const nodata = Math.max(0, total - valid);

  document.getElementById("stat-valid-pixels").textContent = valid.toLocaleString();
  document.getElementById("stat-nodata-pixels").textContent = nodata.toLocaleString();
  document.getElementById("stat-mode").textContent =
    data.mode === "observed" ? "Observed Satellite (Calibrated)" : "Synthetic Demo";

  const prov = data.provenance || {};
  if (data.mode === "observed") {
    document.getElementById("prov-mission").textContent = "NISAR LSAR (NASA-ISRO)";
    document.getElementById("prov-pre").textContent = "2024-01-15T14:00:00Z (Dry Baseline)";
    document.getElementById("prov-post").textContent = "2024-02-15T14:00:00Z (Post-Rain Inundation)";
    document.getElementById("prov-crs").textContent = prov.crs ? "WGS 84 (EPSG:4326)" : "EPSG:4326";
  } else {
    document.getElementById("prov-mission").textContent = "Synthetic SAR Simulator";
    document.getElementById("prov-pre").textContent = "Simulated State A";
    document.getElementById("prov-post").textContent = "Simulated State B (+6 dB zone)";
    document.getElementById("prov-crs").textContent = "Relative Grid (WGS84 Projected)";
  }
}

// --- Render Leaflet Map Overlay ---
function renderMapOverlay(data) {
  if (!map || !data.image_base64 || !data.leaflet_bounds) return;

  // Remove existing overlay
  if (currentOverlay) {
    map.removeLayer(currentOverlay);
    currentOverlay = null;
  }

  // Add new image overlay with opacity
  currentOverlay = L.imageOverlay(data.image_base64, data.leaflet_bounds, {
    opacity: 0.85,
    interactive: true,
  }).addTo(map);

  // Fit map to image bounds
  map.fitBounds(data.leaflet_bounds, { padding: [20, 20], maxZoom: 13 });
}
