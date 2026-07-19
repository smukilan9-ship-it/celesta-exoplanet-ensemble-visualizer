# Celesta Exoplanet — Ensemble Visualizer

An exact animated trace of the final LightGBM + CatBoost ensemble.

Choose a real held-out KOI and watch the complete path: 140 raw fields, 98 allowed measurements, 1,350 LightGBM trees, 450 CatBoost trees, two probability vectors, the locked weighted blend, and the final class.

## Run locally

```bash
npm ci
npm run check
python3 -m http.server 8004
```

Open <http://127.0.0.1:8004/>. Add `?embed=1` for the responsive embedded mode used by the flagship app.

## What is exact

- Every object is a real out-of-fold example.
- Leaf indices come from the trained members used for that held-out fold.
- The visual shows all 1,800 trees, not decorative stand-ins.
- The probability vectors and weights match the selected ensemble.

The visual explains one decision. The [reproducible model repository](https://github.com/smukilan9-ship-it/celesta-exoplanet-reproducible-model) rebuilds the complete 9,564-row result.
