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

The demo attempts:
- `POST /recommend`

If backend is not available, it automatically shows a mock response.

## Pending UX improvements

- Add a map picker modal for start location:
  - click button next to lat/lon
  - open mini-map centered on selected city
  - user clicks a point
  - lat/lon fields auto-update
  - modal closes and returns to main form

## Files

- `frontend/index.html`
- `frontend/styles.css`
- `frontend/app.js`
