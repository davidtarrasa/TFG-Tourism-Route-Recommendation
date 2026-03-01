const CITY_META = {
  Q35765: { name: "Osaka", center: [34.6937, 135.5023] },
  Q406: { name: "Istanbul", center: [41.0082, 28.9784] },
  Q864965: { name: "Petaling Jaya", center: [3.1073, 101.6067] },
};

const CATEGORY_POOL = [
  "food",
  "culture",
  "nature",
  "nightlife",
  "shopping",
  "service",
  "health",
  "entertainment",
  "transport",
  "relaxation",
  "family",
  "sports",
];
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

function parseCoord(value) {
  if (value == null) return null;
  const s = String(value).trim().replace(",", ".");
  if (!s) return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

let map;
let markersLayer;
let routeLayer;
let currentResult = null;

let mapPicker;
let mapPickerMarker;
let pickerLatLon = null;

const form = document.getElementById("recommend-form");
const cityInput = document.getElementById("city");
const stopsInput = document.getElementById("stops");
const stopsValue = document.getElementById("stops-value");
const generateBtn = document.getElementById("generate-btn");
const exportBtn = document.getElementById("export-btn");
const saveBtn = document.getElementById("save-btn");
const resetSavedBtn = document.getElementById("reset-saved-btn");
const refreshSavedBtn = document.getElementById("refresh-saved-btn");
const errorState = document.getElementById("status-error");
const infoState = document.getElementById("status-info");
const loadingState = document.getElementById("status-loading");
const poiList = document.getElementById("poi-list");
const resultMeta = document.getElementById("result-meta");
const mapCaption = document.getElementById("map-caption");
const routeVariants = document.getElementById("route-variants");
const routeSummary = document.getElementById("route-summary");
const savedRoutesList = document.getElementById("saved-routes-list");
const latInput = document.getElementById("lat");
const lonInput = document.getElementById("lon");
const openMapPickerBtn = document.getElementById("open-map-picker-btn");
const closeMapPickerBtn = document.getElementById("close-map-picker-btn");
const applyMapPickerBtn = document.getElementById("apply-map-picker-btn");
const mapPickerModal = document.getElementById("map-picker-modal");
const mapPickerBackdrop = document.getElementById("map-picker-backdrop");
const mapPickerCaption = document.getElementById("map-picker-caption");

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

function applyCityCenter(qid, force = true) {
  const c = CITY_META[qid];
  if (!c) return;
  const latStr = Number(c.center[0]).toFixed(4);
  const lonStr = Number(c.center[1]).toFixed(4);
  if (force || !latInput.value) latInput.value = latStr;
  if (force || !lonInput.value) lonInput.value = lonStr;

  if (mapPicker && !mapPickerModal.classList.contains("hidden")) {
    mapPicker.setView(c.center, 13);
  }
}

function buildPayload() {
  const cityQid = cityInput.value;
  const budget = document.getElementById("budget").value;
  const prefs = readPreferences();
  const proximity = document.getElementById("proximity").checked;
  const strictRealMode = document.getElementById("strict-real-mode").checked;
  const userIdRaw = document.getElementById("user-id").value;

  const lat = parseCoord(latInput.value);
  const lon = parseCoord(lonInput.value);
  const userId = userIdRaw ? Number(userIdRaw) : null;

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
    _prefer_location: proximity,
    _strict_real_mode: strictRealMode,
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
    payload.lat != null && payload.lon != null ? [payload.lat, payload.lon] : city.center;
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

async function fetchRecommendation(payload, allowMock = true) {
  const api = apiBaseUrl();
  const requestPayload = { ...payload };
  delete requestPayload._prefer_location;
  delete requestPayload._strict_real_mode;
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
    if (!allowMock) throw err;
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

  map.fitBounds(L.latLngBounds(latlngs).pad(0.2));

  L.polyline(latlngs, { color: "#0a84ff", weight: 5, opacity: 0.9 }).addTo(routeLayer);

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
      ? (rows.reduce((acc, r) => acc + (Number(r.rating) || 0), 0) / rows.length).toFixed(2)
      : "--";
    const totalDist = rows.reduce((acc, r) => acc + (Number(r.distance_km) || 0), 0);
    const card = document.createElement("article");
    card.className = `route-summary-card ${key === activeKey ? "is-active" : ""}`;
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
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function renderSavedRoutes(items, source) {
  savedRoutesList.innerHTML = "";
  if (!items.length) {
    savedRoutesList.innerHTML = `<p class="poi-meta">No hay rutas guardadas (${source}).</p>`;
    return;
  }
  items.forEach((it) => {
    const routeType = it.route_type || it.selectedVariant || "route";
    const created = it.created_at || it.saved_at || "";
    const city = it.city_qid || it.city || "n/a";
    const user = it.user_id ?? "anon";
    const row = document.createElement("article");
    row.className = "saved-item";
    row.innerHTML = `
      <h5>${routeType} · ${city}</h5>
      <p>user: ${user} · ${String(created).replace("T", " ").slice(0, 19)} · fuente: ${source}</p>
    `;
    savedRoutesList.appendChild(row);
  });
}

async function loadSavedRoutes() {
  const cityQid = cityInput.value || "";
  const userIdRaw = document.getElementById("user-id").value;
  const userId = userIdRaw ? Number(userIdRaw) : null;
  const api = apiBaseUrl();
  const query = new URLSearchParams();
  query.set("limit", "30");
  if (cityQid) query.set("city_qid", cityQid);
  if (userId != null) query.set("user_id", String(userId));
  try {
    const response = await fetch(`${api}/saved-routes?${query.toString()}`);
    if (response.ok) {
      const data = await response.json();
      renderSavedRoutes(data.items || [], "backend");
      return;
    }
  } catch (_) {}
  const local = JSON.parse(localStorage.getItem("tfg_saved_routes") || "[]");
  const filtered = local.filter((x) => (!cityQid || x.city_qid === cityQid) && (userId == null || x.user_id === userId));
  renderSavedRoutes(filtered.slice().reverse().slice(0, 30), "local");
}

async function saveCurrentResultToBackend(item) {
  const api = apiBaseUrl();
  const routePayload = currentResult?.routes?.[currentResult.selectedVariant] || {};
  const body = {
    user_id: item.user_id ?? null,
    city_qid: item.city_qid ?? null,
    route_type: item.selectedVariant,
    source: "frontend",
    payload: {
      city: item.city,
      selectedVariant: item.selectedVariant,
      route: routePayload,
      full_response: item.payload,
      saved_at: item.saved_at,
    },
  };
  const response = await fetch(`${api}/saved-routes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Backend ${response.status}: ${text}`);
  }
  const data = await response.json();
  setInfo(`Ruta guardada en localStorage + backend (id ${data.id}).`);
  await loadSavedRoutes();
}

function saveCurrentResult() {
  if (!currentResult) return;
  const key = "tfg_saved_routes";
  const prev = JSON.parse(localStorage.getItem(key) || "[]");
  const item = {
    saved_at: new Date().toISOString(),
    city: currentResult.city,
    city_qid: currentResult.cityQid,
    user_id: currentResult.userId,
    selectedVariant: currentResult.selectedVariant,
    payload: currentResult.raw,
  };
  prev.push(item);
  localStorage.setItem(key, JSON.stringify(prev));
  setInfo(`Ruta guardada en localStorage (${prev.length} guardadas). Guardando también en backend...`);
  saveCurrentResultToBackend(item).catch((err) => {
    setInfo(`Guardado local OK. Backend save falló: ${err.message}`);
  });
}

async function resetSavedRoutes() {
  const ok = window.confirm("¿Borrar rutas guardadas? Se limpiará localStorage y backend (si está disponible).");
  if (!ok) return;
  localStorage.removeItem("tfg_saved_routes");

  const cityQid = cityInput.value || null;
  const userIdRaw = document.getElementById("user-id").value;
  const userId = userIdRaw ? Number(userIdRaw) : null;
  const api = apiBaseUrl();
  let deleted = null;
  try {
    const response = await fetch(`${api}/saved-routes`, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, city_qid: cityQid }),
    });
    if (response.ok) {
      const data = await response.json();
      deleted = data.deleted;
    }
  } catch (_) {}

  if (deleted == null) {
    setInfo("LocalStorage reseteado. Backend no disponible o sin borrar.");
  } else {
    setInfo(`LocalStorage reseteado y backend limpiado (${deleted} registros).`);
  }
  await loadSavedRoutes();
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

function openMapPicker() {
  if (!mapPickerModal) {
    setError("No se encontró el modal de mapa en el DOM.");
    return;
  }
  const city = CITY_META[cityInput.value] || CITY_META.Q35765;
  mapPickerModal.classList.remove("hidden");
  mapPickerModal.setAttribute("aria-hidden", "false");

  if (!mapPicker) {
    mapPicker = L.map("map-picker", { zoomControl: true }).setView(city.center, 13);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(mapPicker);
    mapPicker.on("click", (e) => {
      pickerLatLon = [e.latlng.lat, e.latlng.lng];
      if (!mapPickerMarker) {
        mapPickerMarker = L.marker([e.latlng.lat, e.latlng.lng]).addTo(mapPicker);
      } else {
        mapPickerMarker.setLatLng([e.latlng.lat, e.latlng.lng]);
      }
      mapPickerCaption.textContent = `Seleccionado: ${e.latlng.lat.toFixed(6)}, ${e.latlng.lng.toFixed(6)}`;
    });
  }

  mapPicker.setView(city.center, 13);
  const lat = parseCoord(latInput.value);
  const lon = parseCoord(lonInput.value);
  if (lat != null && lon != null) {
    pickerLatLon = [lat, lon];
    if (!mapPickerMarker) mapPickerMarker = L.marker([lat, lon]).addTo(mapPicker);
    else mapPickerMarker.setLatLng([lat, lon]);
    mapPickerCaption.textContent = `Seleccionado: ${lat.toFixed(6)}, ${lon.toFixed(6)}`;
  }
  setTimeout(() => mapPicker.invalidateSize(), 30);
}

function closeMapPicker() {
  mapPickerModal.classList.add("hidden");
  mapPickerModal.setAttribute("aria-hidden", "true");
}

function applyMapPickerSelection() {
  if (pickerLatLon) {
    latInput.value = pickerLatLon[0].toFixed(6);
    lonInput.value = pickerLatLon[1].toFixed(6);
  }
  closeMapPicker();
}

stopsInput.addEventListener("input", () => {
  stopsValue.textContent = stopsInput.value;
});

cityInput.addEventListener("change", () => {
  applyCityCenter(cityInput.value, true);
  loadSavedRoutes().catch(() => {});
});
cityInput.addEventListener("input", () => applyCityCenter(cityInput.value, true));

if (openMapPickerBtn) openMapPickerBtn.addEventListener("click", openMapPicker);
if (closeMapPickerBtn) closeMapPickerBtn.addEventListener("click", closeMapPicker);
if (mapPickerBackdrop) mapPickerBackdrop.addEventListener("click", closeMapPicker);
if (applyMapPickerBtn) applyMapPickerBtn.addEventListener("click", applyMapPickerSelection);
window.openMapPickerUI = openMapPicker;

refreshSavedBtn.addEventListener("click", () => {
  loadSavedRoutes().catch((err) => setError(`No se pudieron cargar rutas guardadas: ${err.message}`));
});

exportBtn.addEventListener("click", () => {
  if (!currentResult) return;
  const cityName = currentResult.city.toLowerCase().replace(/\s+/g, "_");
  downloadJSON(currentResult.raw, `multi_route_${cityName}.json`);
});

saveBtn.addEventListener("click", saveCurrentResult);
resetSavedBtn.addEventListener("click", () => {
  resetSavedRoutes().catch((err) => setError(`Reset falló: ${err.message}`));
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setLoading(true);
  setError("");
  setInfo("");

  const payload = buildPayload();
  let apiResult;
  try {
    apiResult = await fetchRecommendation(payload, !payload._strict_real_mode);
  } catch (err) {
    setLoading(false);
    setError(`Backend no disponible (${err.message}). Estás en modo real: no se usa mock.`);
    return;
  }
  const { data, source, warning } = apiResult;
  const routes = data?.routes || {};
  const selectedVariant = selectPrimaryRoute(routes, payload._prefer_location);
  if (!selectedVariant) {
    setLoading(false);
    setError("No se han generado rutas para esta petición.");
    return;
  }

  currentResult = {
    city: cityNameFromQid(payload.city_qid),
    cityQid: payload.city_qid,
    userId: payload.user_id ?? null,
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
applyCityCenter(cityInput.value, true);
setTimeout(() => applyCityCenter(cityInput.value, true), 0);
window.addEventListener("pageshow", () => applyCityCenter(cityInput.value, true));
loadSavedRoutes().catch(() => {});
