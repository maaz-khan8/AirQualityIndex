## AQI Project – Gap Analysis vs `Key_Features.md`

This document compares the current implementation to the items listed in `Key_Features.md`, highlights what’s remaining, and proposes an ideal, minimal approach to complete them.

### Current Coverage (Done)
- Feature pipeline: Open‑Meteo fetch → feature engineering (lags/rolling/time features) → Hopsworks Feature Store (FG + FV) → hourly retraining via GitHub Actions.
- Model training: Random Forest, Ridge; multi‑horizon (1h,6h,12h,24h,48h,72h); models stored in Hopsworks; local `.pkl` artifacts for dashboard.
- Dashboard (Streamlit): current AQI, trends, multi‑horizon predictions, model comparison, alerts, SHAP explainability, model metrics, data validation, EDA reports.
- CI/CD: GitHub Actions workflow "AQI Daily Update" runs hourly (cron) executing `python -m src.main update`; secrets configured.
- **Backfill CLI**: `python -m src.main setup --start YYYY-MM-DD --end YYYY-MM-DD` with idempotent date-range backfills.
- **SHAP Integration**: TreeExplainer/KernelExplainer analysis with artifacts saved to `artifacts/` and dashboard explainability tab.
- **Alert System**: Threshold-based alerts with severity levels, alert history, and dashboard panel.
- **Registry Metrics**: Sidecar JSON metrics files alongside models with dashboard metrics panel.
- **Data Quality Validation**: Schema, range, null, and freshness checks with validation reports and dashboard panel.
- **EDA Snapshot**: Comprehensive analysis with distributions, correlations, time series, missing data, HTML reports, and dashboard panel.

### Gaps vs Key_Features.md
1) ~~Historical data backfill (robustness)~~ ✅ **COMPLETED**
   - ~~Current: Initial one‑year backfill during `setup` only.~~
   - ~~Gap: Re‑runnable backfill job with idempotency (date‑ranged backfills; avoid duplicates).~~

2) Hourly feature pipeline schedule
   - Current: Configured (hourly GitHub Actions cron). Optional improvement: split ingestion (hourly) and training (daily) into separate workflows to reduce compute.

3) ~~Advanced analytics/EDA~~ ✅ **COMPLETED**
   - ~~Current: No formal EDA module or notebook artifacts.~~
   - ~~Gap: Lightweight EDA report (distribution, seasonality, missingness, drift snapshot) exported to `reports/` and viewable from dashboard link.~~

4) ~~Model interpretability (SHAP) in production path~~ ✅ **COMPLETED**
   - ~~Current: Placeholder; SHAP not executed end‑to‑end during pipeline; no artifacts exposed to dashboard.~~
   - ~~Gap: Batch SHAP computation on latest training window; persist top‑k features and example explanations to Hopsworks/Artifacts; dashboard panel to render.~~

5) ~~Alerting system~~ ✅ **COMPLETED**
   - ~~Current: Placeholder methods; not active.~~
   - ~~Gap: Threshold‑based alert runner with simple notification (email/webhook) and alert history table in Feature Store (or S3/CSV) for dashboard.~~

6) ~~Model registry metrics~~ ✅ **COMPLETED**
   - ~~Current: Model upload succeeds, but `save_metric` not available; metrics not attached to registry objects.~~
   - ~~Gap: Store metrics as sidecar artifact (JSON) alongside model; reference in model description.~~

7) ~~Data quality & monitoring~~ ✅ **COMPLETED**
   - ~~Current: No explicit validation.~~
   - ~~Gap: Add Great Expectations‑style checks (schema, ranges, null %, freshness) pre‑insert and pre‑train; emit summary to logs/artifact.~~

8) Deep learning models (optional in Key_Features)
   - Current: Not implemented.
   - Gap (optional): 1 lightweight baseline (e.g., TemporalConvNet/LSTM) for 6h horizon to compare; behind a flag.

9) API/UI polish
   - Current: Single‑city; timezone assumptions; trends chart improved but basic; no timezone selector.
   - Gap: Add timezone selector; clarify units; allow switching horizons/models; link to EDA report.

### Ideal Minimal Approach to Complete

1) Backfill (idempotent)
- Add CLI: `python -m src.main setup --start YYYY-MM-DD --end YYYY-MM-DD` (non‑breaking default keeps 1‑year).
- Insert with primary keys (`timestamp, city`) and `on_conflict` guard (handled by Hopsworks FG). Log counts.

2) CI scheduling refinement (optional)
- Keep the current hourly workflow as default. If costs/time rise, split into two workflows: (a) hourly ingestion `--ingest-only`; (b) daily training.

3) EDA snapshot
- Add `src/eda.py`: generates a small HTML (pandas‑profiling or custom) and PNG plots; save to `reports/latest/` and upload as workflow artifact.
- Dashboard: add link and last‑generated timestamp.

4) SHAP integration
- During training, compute SHAP (TreeExplainer/KernelExplainer) on a validation slice; write `artifacts/shap_top_features.json` and a few PNGs.
- Dashboard: new “Explainability” tab reading the JSON + images.

5) Alerts
- Implement `src/alerts.py`: evaluate latest AQI and predictions vs thresholds; persist alert rows (timestamp, value, severity, message) to a small FG `aqi_alerts`.
- Optional webhook/email step in CI (config‑driven, can be disabled).

6) Registry metrics
- Save `metrics_{model}.json` alongside each model; include metrics path in model description. Provide loader in dashboard.

7) Data quality checks
- Add `src/validation.py`: schema + range checks; invoked before insert and before train; write summary JSON to `artifacts/validation_summary.json` and fail hard on severe breaches.

8) (Optional) DL baseline
- Add `src/models/deep_learning.py` with a small LSTM/TCN for the 6h horizon; train behind `ENABLE_DL=true` env flag.

### Deliverables & Order (1–2 weeks)
1) ✅ **Backfill CLI** (low risk, high value). Hourly schedule is already enabled.
2) ✅ **SHAP artifacts** in training + dashboard tab.
3) ✅ **Alerts FG** + simple notifier + dashboard panel.
4) ✅ **Registry metrics** as sidecar JSON.
5) ✅ **Data validation** hooks.
6) ✅ **EDA snapshot** + dashboard link.
7) (Optional) DL baseline.

### Notes
- We keep the unified pipeline (`src/pipeline.py`) and expose new modes/flags rather than adding many files.
- We rely on Hopsworks for governance; where APIs lack `save_metric`, we store artifacts next to models.

### Implementation Status Summary
**✅ COMPLETED (6/7 major deliverables):**
- Backfill CLI with date range parameters
- SHAP Integration with TreeExplainer/KernelExplainer
- Alert System with threshold-based notifications
- Registry Metrics with sidecar JSON storage
- Data Quality Validation with comprehensive checks
- EDA Snapshot with HTML reports and visualizations

**🔄 REMAINING (1/7 major deliverables):**
- Deep Learning baseline (optional)

**📊 Current System Capabilities:**
- Hourly automated pipeline with GitHub Actions
- Comprehensive dashboard with 6 analysis panels
- Multi-horizon forecasting (1h to 72h)
- Model interpretability and performance tracking
- Data quality monitoring and validation
- Automated alerting and reporting
- Historical data backfill capabilities


