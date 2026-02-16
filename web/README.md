# Web Demo (Frontend)

One-page responsive demo for:
- request configuration
- route generation UX
- map + POI list visualization

## Run locally

Option 1 (quick):
- Open `web/index.html` directly in the browser.

Option 2 (recommended):
- Serve with a static server from repo root:

```bash
python -m http.server 8081
```

Then open:
- `http://localhost:8081/web/`

## Backend integration

The demo attempts:
- `POST /recommend`

If backend is not available, it automatically shows a mock response.

## Files

- `web/index.html`
- `web/styles.css`
- `web/app.js`
