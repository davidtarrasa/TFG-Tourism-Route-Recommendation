# Frontend Demo

One-page responsive demo for:
- request configuration
- route generation UX
- map + POI list visualization

## Run locally

Option 1 (quick):
- Open `frontend/index.html` directly in the browser.

Option 2 (recommended):
- Serve with a static server from repo root:

```bash
python -m http.server 8081
```

Then open:
- `http://localhost:8081/frontend/`

## Backend integration

The demo uses backend endpoints:
- `POST /multi-recommend`
- `POST /saved-routes`
- `GET /saved-routes`
- `DELETE /saved-routes`

If backend is not available, the UI shows an error (no mock fallback).

## Pending UX improvements

- Add optional browser geolocation button ("Use my current position").

## Files

- `frontend/index.html`
- `frontend/styles.css`
- `frontend/app.js`
