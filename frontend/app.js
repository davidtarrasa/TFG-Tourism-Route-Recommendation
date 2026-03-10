const CITY_META = {
  Q35765: { name: "Osaka", center: [34.6937, 135.5023] },
  Q406: { name: "Istanbul", center: [41.0082, 28.9784] },
  Q864965: { name: "Petaling Jaya", center: [3.1073, 101.6067] },
};

const VARIANT_ORDER = ["full", "history", "inputs", "location"];
const VARIANT_LABEL = { full: "Full", history: "History", inputs: "Inputs", location: "Location" };

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

function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const toRad = (x) => (x * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

let map;
let markersLayer;
let routeLayer;
let currentResult = null;
let mapRenderSeq = 0;
let segmentLayers = { straight: [], road: [], visible: [] };
let satelliteLayer;
let lightLayer;
let osmLayer;
let currentBaseLayer;
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
const mapCard = document.getElementById("map-card");
const mapStyleSelect = document.getElementById("map-style");
const mapFullscreenBtn = document.getElementById("map-fullscreen-btn");
const routeLegend = document.getElementById("route-legend");
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
  satelliteLayer = L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    { maxZoom: 19, attribution: "Esri" }
  );
  lightLayer = L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    maxZoom: 20,
    attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
  });
  osmLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors",
  });
  currentBaseLayer = satelliteLayer;
  currentBaseLayer.addTo(map);
  markersLayer = L.layerGroup().addTo(map);
  routeLayer = L.layerGroup().addTo(map);
}

function setBaseLayer(style) {
  const wanted =
    style === "light" ? lightLayer : style === "osm" ? osmLayer : satelliteLayer;
  if (!wanted || !map) return;
  if (currentBaseLayer && map.hasLayer(currentBaseLayer)) map.removeLayer(currentBaseLayer);
  wanted.addTo(map);
  currentBaseLayer = wanted;
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
  if (force || !latInput.value) latInput.value = Number(c.center[0]).toFixed(4);
  if (force || !lonInput.value) lonInput.value = Number(c.center[1]).toFixed(4);
  if (mapPicker && !mapPickerModal.classList.contains("hidden")) mapPicker.setView(c.center, 13);
}

function buildPayload() {
  const cityQid = cityInput.value;
  const budget = document.getElementById("budget").value;
  const prefs = readPreferences();
  const proximity = document.getElementById("proximity").checked;
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
    visits_limit: 10000,
    build_route: true,
    _prefer_location: proximity,
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
  const response = await fetch(`${api}/multi-recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestPayload),
  });
  if (!response.ok) throw new Error(`Backend ${response.status}: ${await response.text()}`);
  const data = await response.json();
  return { data, source: "backend", warning: "" };
}

function routeWithOrder(route) {
  const orderedRows = route?.route?.ordered_pois;
  const rawRows = route?.results;
  const rows = Array.isArray(orderedRows) && orderedRows.length ? orderedRows : rawRows || [];
  return rows.map((p, idx) => ({ ...p, order: idx + 1 }));
}

function withLegDistances(pois) {
  return pois.map((poi, i) => {
    if (Number.isFinite(Number(poi.distance_km))) return poi;
    if (i === 0) return { ...poi, distance_km: null, leg_distance_km: null };
    const prev = pois[i - 1];
    const ok =
      Number.isFinite(Number(prev?.lat)) &&
      Number.isFinite(Number(prev?.lon)) &&
      Number.isFinite(Number(poi.lat)) &&
      Number.isFinite(Number(poi.lon));
    if (!ok) return { ...poi, distance_km: null, leg_distance_km: null };
    return {
      ...poi,
      leg_distance_km: haversineKm(Number(prev.lat), Number(prev.lon), Number(poi.lat), Number(poi.lon)),
    };
  });
}

function routeTotalKm(route, pois) {
  const backendTotal = Number(route?.route?.total_km);
  if (Number.isFinite(backendTotal) && backendTotal > 0) return backendTotal;
  let total = 0;
  for (let i = 1; i < pois.length; i += 1) {
    const d = Number(pois[i].distance_km ?? pois[i].leg_distance_km);
    if (Number.isFinite(d)) total += d;
  }
  return total;
}

function segmentColorBlue(idx, total) {
  const n = Math.max(total - 1, 1);
  const t = idx / n;
  const hue = 220 - Math.round(45 * t);
  const sat = 90;
  const light = 45 + Math.round(12 * t);
  return `hsl(${hue}, ${sat}%, ${light}%)`;
}

function segmentColorWarm(idx) {
  const warm = ["#ff6d00", "#ff7043", "#ff5722", "#f4511e"];
  return warm[idx % warm.length];
}

async function osrmLeg(start, end) {
  const [lat1, lon1] = start;
  const [lat2, lon2] = end;
  const url =
    `https://router.project-osrm.org/route/v1/driving/${lon1},${lat1};${lon2},${lat2}` +
    "?overview=full&geometries=geojson";
  const r = await fetch(url);
  if (!r.ok) return null;
  const payload = await r.json();
  const coords = payload?.routes?.[0]?.geometry?.coordinates || [];
  if (!coords.length) return null;
  return coords.map((c) => [Number(c[1]), Number(c[0])]);
}

async function drawRoadRoute(latlngs, renderSeq) {
  const legSegments = [];
  for (let i = 0; i < latlngs.length - 1; i += 1) {
    try {
      const leg = await osrmLeg(latlngs[i], latlngs[i + 1]);
      if (leg && leg.length) legSegments.push({ idx: i, coords: leg });
      else legSegments.push({ idx: i, coords: [latlngs[i], latlngs[i + 1]] });
    } catch (_) {
      legSegments.push({ idx: i, coords: [latlngs[i], latlngs[i + 1]] });
    }
  }
  if (renderSeq !== mapRenderSeq) return;
  legSegments.forEach((seg) => {
    const line = segmentLayers.road[seg.idx];
    if (!line) return;
    line.setLatLngs(seg.coords);
  });
}

function renderLegend(latlngs) {
  if (!routeLegend) return;
  const legs = Math.max(latlngs.length - 1, 0);
  if (!legs) {
    routeLegend.classList.add("hidden");
    routeLegend.innerHTML = "";
    return;
  }
  routeLegend.classList.remove("hidden");
  const maxShow = legs;
  const rows = [];
  for (let i = 0; i < maxShow; i += 1) {
    const checked = segmentLayers.visible[i] !== false ? "checked" : "";
    rows.push(
      `<label class="route-legend-row"><input class="route-legend-check" type="checkbox" data-seg="${i}" ${checked} /><span class="route-legend-color" style="background:${segmentColorBlue(
        i,
        legs
      )}"></span><span>Tramo ${i + 1}</span></label>`
    );
  }
  routeLegend.innerHTML = `<div class="route-legend-title">Orden de tramos (azul = ruta calles)</div>${rows.join(
    ""
  )}`;
  routeLegend.querySelectorAll(".route-legend-check").forEach((el) => {
    el.addEventListener("change", (ev) => {
      const idx = Number(ev.target.dataset.seg);
      const on = !!ev.target.checked;
      segmentLayers.visible[idx] = on;
      const a = segmentLayers.straight[idx];
      const b = segmentLayers.road[idx];
      [a, b].forEach((layer) => {
        if (!layer) return;
        if (on) routeLayer.addLayer(layer);
        else routeLayer.removeLayer(layer);
      });
    });
  });
}

function renderMap(pois, cityName, variant) {
  mapRenderSeq += 1;
  const renderSeq = mapRenderSeq;
  markersLayer.clearLayers();
  routeLayer.clearLayers();
  if (!pois.length) {
    mapCaption.textContent = "Sin POIs para dibujar";
    if (routeLegend) {
      routeLegend.classList.add("hidden");
      routeLegend.innerHTML = "";
    }
    return;
  }
  const latlngs = pois
    .filter((p) => Number.isFinite(Number(p.lat)) && Number.isFinite(Number(p.lon)))
    .map((p) => [Number(p.lat), Number(p.lon)]);
  if (!latlngs.length) {
    mapCaption.textContent = "Sin coordenadas disponibles en esta variante";
    if (routeLegend) {
      routeLegend.classList.add("hidden");
      routeLegend.innerHTML = "";
    }
    return;
  }
  map.fitBounds(L.latLngBounds(latlngs).pad(0.2));
  const legs = Math.max(latlngs.length - 1, 0);
  segmentLayers = {
    straight: new Array(legs).fill(null),
    road: new Array(legs).fill(null),
    visible: new Array(legs).fill(true),
  };

  for (let i = 0; i < latlngs.length - 1; i += 1) {
    const line = L.polyline([latlngs[i], latlngs[i + 1]], {
      color: segmentColorWarm(i),
      weight: 2.8,
      opacity: 0.95,
      dashArray: "10,6",
    });
    segmentLayers.straight[i] = line;
    if (segmentLayers.visible[i] !== false) {
      line.addTo(routeLayer);
    }

    // Fallback inmediato para ruta por calles (se reemplaza luego por OSRM).
    const roadLine = L.polyline([latlngs[i], latlngs[i + 1]], {
      color: segmentColorBlue(i, latlngs.length - 1),
      weight: 4,
      opacity: 0.98,
    });
    segmentLayers.road[i] = roadLine;
    if (segmentLayers.visible[i] !== false) {
      roadLine.addTo(routeLayer);
    }
  }
  drawRoadRoute(latlngs, renderSeq);
  renderLegend(latlngs);

  pois.forEach((poi) => {
    if (!Number.isFinite(Number(poi.lat)) || !Number.isFinite(Number(poi.lon))) return;
    const icon = L.divIcon({
      className: "order-badge",
      html: `<div style="width:24px;height:24px;border-radius:50%;background:#0b3d91;color:#fff;font-size:12px;font-weight:700;line-height:24px;text-align:center;border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.45);">${poi.order}</div>`,
      iconSize: [24, 24],
      iconAnchor: [12, 12],
    });
    const marker = L.marker([Number(poi.lat), Number(poi.lon)], { icon }).addTo(markersLayer);
    marker.bindTooltip(`${poi.order}. ${poi.name}`, { direction: "top" });
    marker.bindPopup(
      `<strong>${poi.order}. ${poi.name}</strong><br/>${poi.primary_category || "N/A"} · Rating: ${poi.rating ?? "N/A"}`
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
    const dist = Number.isFinite(Number(poi.distance_km))
      ? Number(poi.distance_km)
      : Number.isFinite(Number(poi.leg_distance_km))
      ? Number(poi.leg_distance_km)
      : null;
    const item = document.createElement("article");
    item.className = "poi-item";
    item.innerHTML = `
      <h4>${poi.order}. ${poi.name}</h4>
      <div class="poi-meta">
        <span>Categoría: ${poi.primary_category ?? "N/A"}</span>
        <span>Rating: ${poi.rating ?? "N/A"}</span>
        <span>Distancia: ${dist == null ? "N/A" : dist.toFixed(2)} km</span>
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
    const n = routeWithOrder(variants[key]).length;
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
    const rows = withLegDistances(routeWithOrder(variants[key]));
    const avgRating = rows.length
      ? (rows.reduce((acc, r) => acc + (Number(r.rating) || 0), 0) / rows.length).toFixed(2)
      : "--";
    const totalDist = routeTotalKm(variants[key], rows);
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
  if (!response.ok) throw new Error(`Backend ${response.status}: ${await response.text()}`);
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
    setInfo(`Guardado local OK. Backend save fallo: ${err.message}`);
  });
}

function toggleMapFullscreen() {
  const card = mapCard || document.querySelector(".map-card");
  if (!card) return;
  if (!document.fullscreenElement) {
    card.requestFullscreen?.();
  } else {
    document.exitFullscreen?.();
  }
}

function onFullscreenChanged() {
  if (mapFullscreenBtn) {
    mapFullscreenBtn.textContent = document.fullscreenElement ? "Exit" : "Fullscreen";
  }
  setTimeout(() => map?.invalidateSize(), 100);
}

async function resetSavedRoutes() {
  const ok = window.confirm("¿Borrar rutas guardadas? Se limpiara localStorage y backend (si esta disponible).");
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
    if (response.ok) deleted = (await response.json()).deleted;
  } catch (_) {}
  if (deleted == null) setInfo("LocalStorage reseteado. Backend no disponible o sin borrar.");
  else setInfo(`LocalStorage reseteado y backend limpiado (${deleted} registros).`);
  await loadSavedRoutes();
}

function cityNameFromQid(qid) {
  return CITY_META[qid]?.name || qid || "Unknown";
}

function renderSelectedVariant(source) {
  if (!currentResult) return;
  const selected = currentResult.selectedVariant;
  const route = currentResult.routes[selected];
  const pois = withLegDistances(routeWithOrder(route));
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
    setError("No se encontro el modal de mapa en el DOM.");
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
      if (!mapPickerMarker) mapPickerMarker = L.marker([e.latlng.lat, e.latlng.lng]).addTo(mapPicker);
      else mapPickerMarker.setLatLng([e.latlng.lat, e.latlng.lng]);
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
if (mapStyleSelect) mapStyleSelect.addEventListener("change", (e) => setBaseLayer(e.target.value));
if (mapFullscreenBtn) mapFullscreenBtn.addEventListener("click", toggleMapFullscreen);
document.addEventListener("fullscreenchange", onFullscreenChanged);
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
  resetSavedRoutes().catch((err) => setError(`Reset fallo: ${err.message}`));
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setLoading(true);
  setError("");
  setInfo("");
  const payload = buildPayload();
  let apiResult;
  try {
    apiResult = await fetchRecommendation(payload);
  } catch (err) {
    setLoading(false);
    setError(`Backend no disponible (${err.message}).`);
    return;
  }

  const { data, source, warning } = apiResult;
  const routes = data?.routes || {};
  const selectedVariant = selectPrimaryRoute(routes, payload._prefer_location);
  if (!selectedVariant) {
    setLoading(false);
    setError("No se han generado rutas para esta peticion.");
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
