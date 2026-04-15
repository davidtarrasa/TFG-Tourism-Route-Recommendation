# Frontend Demo (Product UI Prototype)

Single-page responsive interface for the tourism route recommender.

Current role:

- consume real backend (`/multi-recommend`)
- render route variants on map
- show POI list and route summaries
- allow saving and listing generated routes

No mock fallback is enabled now:

- if backend is down, UI shows backend error state

## Run

From repo root:

```bash
python -m http.server 8081
```

Open:

- `http://localhost:8081/frontend/`

Backend must be running (default expected base URL):

- `http://localhost:8000`

You can override API URL with query param:

- `http://localhost:8081/frontend/?api=http://127.0.0.1:8000`

## Backend endpoints used by UI

- `POST /multi-recommend`
- `POST /saved-routes`
- `GET /saved-routes`
- `DELETE /saved-routes`

## Main UI Features

- City selector (`Q35765`, `Q406`, `Q864965`)
- Stops slider (`k`)
- Preferences chips (intent-level categories)
- Budget + proximity toggle
- Optional `user_id`
- `lat/lon` auto-filled to city center
- Map picker modal to select coordinates on map
- Route generation button
- Variant tabs/cards:
  - `Full`
  - `History`
  - `Inputs`
  - `Location`
- Map style selector (Satellite / Light / OSM)
- Fullscreen map button
- Segment legend with per-leg show/hide checkboxes
- POI list with order, category, rating, distance
- Export JSON
- Save route
- Saved-routes panel with refresh + reset

## Route rendering behavior

For selected variant:

- numbered circular markers (`1..N`)
- straight dashed edges (visual reference)
- road route overlay (OSRM from frontend request)
- per-segment color cascade
- caption with city, variant, and stops count

## Saved routes behavior

When user clicks Save:

1. always saved in `localStorage`
2. then backend save is attempted

Saved list load:

- tries backend first
- if backend unavailable, falls back to localStorage list

Reset saved:

- clears localStorage
- tries backend `DELETE /saved-routes` with current city/user filters

## Signals and warnings shown in UI

Frontend reads backend `omitted` and `warnings`:

- if history route is omitted due to missing city history, user sees explicit info message
- other backend warnings are also displayed

## Files

- `frontend/index.html`
- `frontend/styles.css`
- `frontend/app.js`
