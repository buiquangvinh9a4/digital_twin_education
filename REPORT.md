# Digital Twin Report — OULAD Learning Analytics

## 1. Executive Summary
- Goal: Build a Digital Twin for a university course to monitor, simulate, and predict student performance using OULAD.
- Twin Concept: Mirror the physical learning process (LMS interactions and outcomes) with a digital counterpart that can simulate interventions and provide early warnings.
- Outcomes:
  - Unified dashboards for class overview and per-student analysis.
  - Realtime-like simulation stream to stress-test monitoring and alerting.
  - Early-warning workflow to identify at-risk students and suggest interventions.

## 2. Digital Twin Architecture
- Physical Twin (Reality):
  - Source: Processed OULAD data (`data/processed/test.csv`, `ou_real.csv`).
  - Signals: clicks, submissions, average score, completion ratio, active weeks, final label.
- Digital Twin (Simulation/Model):
  - Predictive layer: LSTM model (`models/oulad_lstm.h5`) when available; fallback logistic-like simulation.
  - Scenario engine: `scripts/simulate_realtime.py` updates `data/simulations/metrics_log.csv` and `status.json`.
  - Watcher: `scripts/update_twin.py` refreshes predictions when inputs change.
- Visualization/Control: Streamlit app (`app/`) with pages for overview, student twin, early warning, class list, and live monitor.

## 3. Data Flow
1. ETL prepares processed data: `scripts/etl_prepare.py` → `data/processed/test.csv` (+ optional `ou_real.csv`, `ou_pred.csv`).
2. Live simulation (optional): `scripts/simulate_realtime.py` writes aggregate metrics and status.
3. Watcher (optional): `scripts/update_twin.py` rebuilds predictions on file changes.
4. Dashboard (`streamlit run app/dashboard.py`) reads both physical and digital sources for side-by-side analytics.

## 4. Key Twin Features in the UI
- 1️⃣ Class Overview (`app/pages/1_Class_Overview.py`)
  - Side-by-side charts of Physical vs Digital metrics and Twin Delta radar.
  - Sliders to simulate changes and recompute a composite Learning Index.
- 2️⃣ Student Twin (`app/pages/2_Student_Twin.py`)
  - Per-student what-if inputs; LSTM prediction if available, otherwise calibrated synthetic score.
  - Twin Delta table, radar, weekly trend comparison; logs simulation history.
- 3️⃣ Early Warning (`app/pages/3_Early_Warning.py`)
  - Risk detection using composite index and probability thresholds.
  - Distribution and radar charts; per-student explanation and intervention hints.
- 4️⃣ Class List (`app/pages/4_Class_List.py`)
  - Access to saved simulation history with CSV export.
- 5️⃣ Live Monitor (`app/pages/5_Live_Monitor.py`)
  - Realtime-like time series by scenario with live status badge.

## 5. Composite Learning Index (0–100)
- Weights: Score (0.30), Completion (0.25), Submissions (0.20), Clicks (0.15), Active Weeks (0.10).
- Aggregates heterogeneous signals into a single interpretable indicator for risk and progress.

## 6. Realtime Simulation & Scenarios
- `simulate_realtime.py` cycles scenarios like `exam_season`, `holiday`, `early_intervention`, `late_dropout`.
- Writes rolling aggregates to `data/simulations/metrics_log.csv` and metadata to `data/simulations/status.json`.
- Dashboard reflects changes via the Live Monitor page and status badges.

## 7. Model Layer
- Preferred: LSTM sequence model (`models/oulad_lstm.h5`), loaded in Student Twin when present.
- Fallback: Sigmoid over a weighted composite score for demos without trained models.

## 8. How to Run (Quickstart)
```bash
cd "/Users/macos/Documents/UET/Các vấn đề hiện đại CNPM/Digital Twin/projects/dt-oulad"
pip install -r requirements.txt
python scripts/etl_prepare.py
# optional terminals:
python scripts/update_twin.py   # terminal A
python scripts/simulate_realtime.py   # terminal B
streamlit run app/dashboard.py  # dashboard
```

## 9. KPIs and Evidence of Twin Value
- Faster insight: unified composite index and radar-based Twin Delta.
- Earlier detection of at-risk students: combined thresholds on index and probability.
- Rapid hypothesis testing: scenario sliders and realtime monitoring enable proactive interventions.

## 10. Limitations & Next Steps
- Data realism: connect to live LMS for production; refine simulation.
- Model robustness: more features, longer sequences, calibration; add explainability.
- Interventions: A/B test recommendations; close the loop with actual outcomes.
- Governance: logging, audit trails, role-based access.
