const CITY_META = {
  Q35765: { name: "Osaka", center: [34.6937, 135.5023] },
  Q406: { name: "Istanbul", center: [41.0082, 28.9784] },
  Q864965: { name: "Petaling Jaya", center: [3.1073, 101.6067] },
};

const CATEGORY_POOL = ["Food", "Culture", "Nature", "Nightlife", "Shopping"];

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
  const prefs = readPreferences();

  return {
    city_qid: cityQid,
    k: Number(stopsInput.value),
    preferences: prefs,
    budget: document.getElementById("budget").value,
    max_price_tier: budgetToTier(document.getElementById("budget").value),
    prioritize_proximity: document.getElementById("proximity").checked,
    lat: latRaw ? Number(latRaw) : null,
    lon: lonRaw ? Number(lonRaw) : null,
    user_id: userIdRaw ? Number(userIdRaw) : null,
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

function makeMockResponse(payload) {
  const city = CITY_META[payload.city_qid] || CITY_META.Q35765;
  const center = payload.lat && payload.lon ? [payload.lat, payload.lon] : city.center;
  const selectedCats = payload.preferences.length ? payload.preferences : CATEGORY_POOL;
  const pois = [];
  let prev = center;

  for (let i = 0; i < payload.k; i += 1) {
    const lat = center[0] + randomBetween(-0.03, 0.03);
    const lon = center[1] + randomBetween(-0.03, 0.03);
    const point = [lat, lon];
    const dist = toKm(prev, point);
    prev = point;
    pois.push({
      order: i + 1,
      fsq_id: `mock_${payload.city_qid}_${i + 1}`,
      name: `POI ${i + 1} · ${city.name}`,
      primary_category: selectedCats[i % selectedCats.length],
      rating: Number(randomBetween(6.4, 9.3).toFixed(2)),
      distance_km: Number(dist.toFixed(2)),
      lat,
      lon,
    });
  }

  return {
    source: "mock",
    city: city.name,
    route: {
      mode: "hybrid",
      pois,
    },
    request: payload,
  };
}

async function fetchRecommendation(payload) {
  try {
    const response = await fetch("/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      throw new Error(`Backend error ${response.status}`);
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

function normalizeResult(data) {
  if (Array.isArray(data?.route?.pois)) {
    return {
      city: data.city || "Unknown",
      routeType: data.route.mode || "hybrid",
      pois: data.route.pois,
    };
  }

  if (Array.isArray(data?.routes) && data.routes.length) {
    const first = data.routes[0];
    return {
      city: data.city || "Unknown",
      routeType: first.type || "route",
      pois: first.pois || [],
    };
  }

  return { city: "Unknown", routeType: "unknown", pois: [] };
}

function renderMap(pois, cityName) {
  markersLayer.clearLayers();
  routeLayer.clearLayers();
  if (!pois.length) return;

  const latlngs = pois.map((p) => [p.lat, p.lon]);
  const bounds = L.latLngBounds(latlngs);
  map.fitBounds(bounds.pad(0.2));

  L.polyline(latlngs, {
    color: "#0a84ff",
    weight: 5,
    opacity: 0.86,
  }).addTo(routeLayer);

  pois.forEach((poi) => {
    const marker = L.circleMarker([poi.lat, poi.lon], {
      radius: 9,
      color: "#0a6bd1",
      fillColor: "#37a0ff",
      fillOpacity: 0.95,
      weight: 2,
    }).addTo(markersLayer);
    marker.bindTooltip(`${poi.order}. ${poi.name}`, { direction: "top" });
    marker.bindPopup(
      `<strong>${poi.order}. ${poi.name}</strong><br/>${poi.primary_category} · Rating: ${
        poi.rating ?? "N/A"
      }`
    );
  });

  mapCaption.textContent = `Ruta en ${cityName} · ${pois.length} paradas`;
}

function renderList(pois, city, source, routeType) {
  poiList.innerHTML = "";
  if (!pois.length) {
    resultMeta.textContent = "Sin resultados";
    return;
  }
  resultMeta.textContent = `${city} · modo ${routeType} · fuente: ${source}`;

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

function downloadJSON(obj, filename) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
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
    routeType: currentResult.routeType,
    payload: currentResult.raw,
  });
  localStorage.setItem(key, JSON.stringify(prev));
  setInfo(`Ruta guardada en localStorage (${prev.length} guardadas).`);
}

stopsInput.addEventListener("input", () => {
  stopsValue.textContent = stopsInput.value;
});

exportBtn.addEventListener("click", () => {
  if (!currentResult) return;
  const cityName = currentResult.city.toLowerCase().replace(/\s+/g, "_");
  downloadJSON(currentResult.raw, `route_${cityName}.json`);
});

saveBtn.addEventListener("click", saveCurrentResult);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setLoading(true);
  setError("");
  setInfo("");

  const payload = buildPayload();
  const { data, source, warning } = await fetchRecommendation(payload);
  const normalized = normalizeResult(data);

  currentResult = {
    city: normalized.city,
    routeType: normalized.routeType,
    raw: data,
  };
  exportBtn.disabled = false;
  saveBtn.disabled = false;

  if (warning) setInfo(warning);
  renderMap(normalized.pois, normalized.city);
  renderList(normalized.pois, normalized.city, source, normalized.routeType);
  setLoading(false);
});

initMap();
