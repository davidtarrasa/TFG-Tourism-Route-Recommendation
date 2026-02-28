const CITY_META = {
  Q35765: { name: "Osaka", center: [34.6937, 135.5023] },
  Q406: { name: "Istanbul", center: [41.0082, 28.9784] },
  Q864965: { name: "Petaling Jaya", center: [3.1073, 101.6067] },
};

const CATEGORY_POOL = ["Food", "Culture", "Nature", "Nightlife", "Shopping"];
const VARIANT_ORDER = ["full", "history", "inputs", "location"];
const VARIANT_LABEL = {
  full: "Full",
  history: "History",
  inputs: "Inputs",
  location: "Location",
};

function apiBaseUrl() {
  const q = new URLSearchParams(window.location.search).get("api");
  if (q) return q.replace(/\/$/, "");
  const { protocol, hostname } = window.location;
  return `${protocol}//${hostname}:8000`;
}

let map;
let markersLayer;
let routeLayer;
let currentResult = null;

const form = document.getElementById("recommend-form");
const stopsInput = document.getElementById("stops");
const stopsValue = document.getElementById("stops-value");
const generateBtn = document.getElementById("generate-btn");
const exportBtn = document.getElementById("export-btn");
const saveBtn = document.getElementById("save-btn");
const errorState = document.getElementById("status-error");
const infoState = document.getElementById("status-info");
const loadingState = document.getElementById("status-loading");
const poiList = document.getElementById("poi-list");
const resultMeta = document.getElementById("result-meta");
const mapCaption = document.getElementById("map-caption");
const routeVariants = document.getElementById("route-variants");
const routeSummary = document.getElementById("route-summary");

function initMap() {
  map = L.map("map", { zoomControl: true }).setView(CITY_META.Q35765.center, 12);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);
  markersLayer = L.layerGroup().addTo(map);
  routeLayer = L.layerGroup().addTo(map);
}

function setLoading(isLoading) {
  generateBtn.disabled = isLoading;
  loadingState.classList.toggle("hidden", !isLoading);
}

function setError(message = "") {
  errorState.textContent = message;
  errorState.classList.toggle("hidden", !message);
}

function setInfo(message = "") {
  infoState.textContent = message;
  infoState.classList.toggle("hidden", !message);
}

function readPreferences() {
  return Array.from(form.querySelectorAll(".checks input:checked")).map((n) => n.value);
}

function budgetToTier(budget) {
  if (budget === "low") return 1;
  if (budget === "medium") return 2;
  return 3;
}

function buildPayload() {
  const cityQid = document.getElementById("city").value;
  const latRaw = document.getElementById("lat").value;
  const lonRaw = document.getElementById("lon").value;
  const userIdRaw = document.getElementById("user-id").value;
  const budget = document.getElementById("budget").value;
  const prefs = readPreferences();
  const proximity = document.getElementById("proximity").checked;

  const lat = latRaw ? Number(latRaw) : null;
  const lon = lonRaw ? Number(lonRaw) : null;
  const userId = userIdRaw ? Number(userIdRaw) : null;

  // Map UI selections into backend prefs string.
  const prefTokens = [...prefs];
  if (budget === "low") prefTokens.push("cheap");
  if (budget === "high") prefTokens.push("expensive");

  return {
    city_qid: cityQid,
    user_id: userId,
    k: Number(stopsInput.value),
    lat,
    lon,
    prefs: prefTokens.join(","),
    max_price_tier: budgetToTier(budget),
    free_only: false,
    category_mode: "soft",
    use_embeddings: true,
    embeddings_path: `src/recommender/cache/word2vec_${cityQid.toLowerCase()}.joblib`,
    use_als: true,
    als_path: `src/recommender/cache/als_${cityQid.toLowerCase()}.joblib`,
    visits_limit: 120000,
    build_route: false,
    // UI-only hint for frontend selection (not sent to backend logic directly).
    _prefer_location: proximity,
  };
}

function randomBetween(min, max) {
  return Math.random() * (max - min) + min;
}

function toKm(a, b) {
  const R = 6371;
  const dLat = ((b[0] - a[0]) * Math.PI) / 180;
  const dLon = ((b[1] - a[1]) * Math.PI) / 180;
  const lat1 = (a[0] * Math.PI) / 180;
  const lat2 = (b[0] * Math.PI) / 180;
  const x =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(x));
}

function makeMockRoute(payload, type) {
  const city = CITY_META[payload.city_qid] || CITY_META.Q35765;
  const center =
    payload.lat && payload.lon ? [payload.lat, payload.lon] : city.center;
  const selectedCats = payload.prefs ? payload.prefs.split(",") : CATEGORY_POOL;
  const pois = [];
  let prev = center;
  for (let i = 0; i < payload.k; i += 1) {
    const lat = center[0] + randomBetween(-0.03, 0.03);
    const lon = center[1] + randomBetween(-0.03, 0.03);
    const point = [lat, lon];
    const dist = toKm(prev, point);
    prev = point;
    pois.push({
      fsq_id: `mock_${payload.city_qid}_${type}_${i + 1}`,
      name: `POI ${i + 1} · ${city.name}`,
      primary_category: selectedCats[i % selectedCats.length] || "Culture",
      rating: Number(randomBetween(6.4, 9.3).toFixed(2)),
      distance_km: Number(dist.toFixed(2)),
      lat,
      lon,
    });
  }
  return { results: pois };
}

function makeMockResponse(payload) {
  return {
    signals: {
      history: !!payload.user_id,
      inputs: !!payload.prefs,
      location: payload.lat != null && payload.lon != null,
    },
    user_exists: !!payload.user_id,
    omitted: {},
    warnings: ["mock_response"],
    routes: {
      full: makeMockRoute(payload, "full"),
      history: makeMockRoute(payload, "history"),
      inputs: makeMockRoute(payload, "inputs"),
      location: makeMockRoute(payload, "location"),
    },
  };
}

function selectPrimaryRoute(routes, preferLocation) {
  const keys = Object.keys(routes || {});
  if (!keys.length) return null;
  if (preferLocation && routes.location) return "location";
  return VARIANT_ORDER.find((k) => routes[k]) || keys[0];
}

async function fetchRecommendation(payload) {
  const api = apiBaseUrl();
  const requestPayload = { ...payload };
  delete requestPayload._prefer_location;
  try {
    const response = await fetch(`${api}/multi-recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestPayload),
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Backend ${response.status}: ${text}`);
    }
    const data = await response.json();
    return { data, source: "backend", warning: "" };
  } catch (err) {
    return {
      data: makeMockResponse(payload),
      source: "mock",
      warning: `Backend no disponible (${err.message}). Mostrando datos mock.`,
    };
  }
}

function routeWithOrder(route) {
  const rows = route?.results || [];
  return rows.map((p, idx) => ({ ...p, order: idx + 1 }));
}

function renderMap(pois, cityName, variant) {
  markersLayer.clearLayers();
  routeLayer.clearLayers();
  if (!pois.length) {
    mapCaption.textContent = "Sin POIs para dibujar";
    return;
  }

  const latlngs = pois
    .filter((p) => p.lat != null && p.lon != null)
    .map((p) => [p.lat, p.lon]);
  if (!latlngs.length) return;

  const bounds = L.latLngBounds(latlngs);
  map.fitBounds(bounds.pad(0.2));

  L.polyline(latlngs, {
    color: "#0a84ff",
    weight: 5,
    opacity: 0.9,
  }).addTo(routeLayer);

  pois.forEach((poi) => {
    if (poi.lat == null || poi.lon == null) return;
    const marker = L.circleMarker([poi.lat, poi.lon], {
      radius: 9,
      color: "#0a6bd1",
      fillColor: "#37a0ff",
      fillOpacity: 0.95,
      weight: 2,
    }).addTo(markersLayer);
    marker.bindTooltip(`${poi.order}. ${poi.name}`, { direction: "top" });
    marker.bindPopup(
      `<strong>${poi.order}. ${poi.name}</strong><br/>${poi.primary_category || "N/A"} · Rating: ${
        poi.rating ?? "N/A"
      }`
    );
  });

  mapCaption.textContent = `${cityName} · ${VARIANT_LABEL[variant] || variant} · ${pois.length} paradas`;
}

function renderList(pois, city, source, routeType) {
  poiList.innerHTML = "";
  if (!pois.length) {
    resultMeta.textContent = "Sin resultados";
    return;
  }
  resultMeta.textContent = `${city} · ruta ${routeType} · fuente: ${source}`;
  pois.forEach((poi) => {
    const item = document.createElement("article");
    item.className = "poi-item";
    item.innerHTML = `
      <h4>${poi.order}. ${poi.name}</h4>
      <div class="poi-meta">
        <span>Categoría: ${poi.primary_category ?? "N/A"}</span>
        <span>Rating: ${poi.rating ?? "N/A"}</span>
        <span>Distancia: ${poi.distance_km ?? "N/A"} km</span>
      </div>
    `;
    poiList.appendChild(item);
  });
}

function renderVariants(variants, activeKey, onSelect) {
  const keys = Object.keys(variants || {});
  if (!keys.length) {
    routeVariants.innerHTML = "";
    routeVariants.classList.add("hidden");
    return;
  }

  routeVariants.classList.remove("hidden");
  routeVariants.innerHTML = "";

  keys.forEach((key) => {
    const n = (variants[key]?.results || []).length;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `variant-btn ${key === activeKey ? "is-active" : ""}`;
    btn.textContent = `${VARIANT_LABEL[key] || key} (${n})`;
    btn.addEventListener("click", () => onSelect(key));
    routeVariants.appendChild(btn);
  });
}

function renderVariantSummary(variants, activeKey, onSelect) {
  const keys = Object.keys(variants || {});
  if (!keys.length) {
    routeSummary.innerHTML = "";
    routeSummary.classList.add("hidden");
    return;
  }

  routeSummary.classList.remove("hidden");
  routeSummary.innerHTML = "";

  keys.forEach((key) => {
    const rows = variants[key]?.results || [];
    const avgRating = rows.length
      ? (
          rows.reduce((acc, r) => acc + (Number(r.rating) || 0), 0) / rows.length
        ).toFixed(2)
      : "--";
    const totalDist = rows.reduce(
      (acc, r) => acc + (Number(r.distance_km) || 0),
      0
    );

    const card = document.createElement("article");
    card.className = `route-summary-card ${
      key === activeKey ? "is-active" : ""
    }`;
    card.innerHTML = `
      <h5>${VARIANT_LABEL[key] || key}</h5>
      <div class="route-summary-meta">
        <span>${rows.length} POIs</span>
        <span>rating medio: ${avgRating}</span>
        <span>dist: ${totalDist.toFixed(2)} km</span>
      </div>
    `;
    card.addEventListener("click", () => onSelect(key));
    routeSummary.appendChild(card);
  });
}

function downloadJSON(obj, filename) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function saveCurrentResult() {
  if (!currentResult) return;
  const key = "tfg_saved_routes";
  const prev = JSON.parse(localStorage.getItem(key) || "[]");
  prev.push({
    saved_at: new Date().toISOString(),
    city: currentResult.city,
    selectedVariant: currentResult.selectedVariant,
    payload: currentResult.raw,
  });
  localStorage.setItem(key, JSON.stringify(prev));
  setInfo(`Ruta guardada en localStorage (${prev.length} guardadas).`);
}

function cityNameFromQid(qid) {
  return CITY_META[qid]?.name || qid || "Unknown";
}

function renderSelectedVariant(source) {
  if (!currentResult) return;
  const selected = currentResult.selectedVariant;
  const route = currentResult.routes[selected];
  const pois = routeWithOrder(route);
  renderMap(pois, currentResult.city, selected);
  renderList(pois, currentResult.city, source, selected);
  renderVariants(currentResult.routes, selected, (nextKey) => {
    currentResult.selectedVariant = nextKey;
    renderSelectedVariant(source);
  });
  renderVariantSummary(currentResult.routes, selected, (nextKey) => {
    currentResult.selectedVariant = nextKey;
    renderSelectedVariant(source);
  });
}

stopsInput.addEventListener("input", () => {
  stopsValue.textContent = stopsInput.value;
});

exportBtn.addEventListener("click", () => {
  if (!currentResult) return;
  const cityName = currentResult.city.toLowerCase().replace(/\s+/g, "_");
  downloadJSON(currentResult.raw, `multi_route_${cityName}.json`);
});

saveBtn.addEventListener("click", saveCurrentResult);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setLoading(true);
  setError("");
  setInfo("");

  const payload = buildPayload();
  const { data, source, warning } = await fetchRecommendation(payload);
  const routes = data?.routes || {};
  const selectedVariant = selectPrimaryRoute(routes, payload._prefer_location);

  if (!selectedVariant) {
    setLoading(false);
    setError("No se han generado rutas para esta petición.");
    return;
  }

  currentResult = {
    city: cityNameFromQid(payload.city_qid),
    selectedVariant,
    routes,
    raw: data,
  };

  exportBtn.disabled = false;
  saveBtn.disabled = false;

  if (warning) setInfo(warning);
  if (Array.isArray(data?.warnings) && data.warnings.length) {
    setInfo(`${warning ? `${warning} · ` : ""}${data.warnings.join(" | ")}`);
  }

  renderSelectedVariant(source);
  setLoading(false);
});

initMap();
