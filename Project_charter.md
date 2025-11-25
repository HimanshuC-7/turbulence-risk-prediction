# Project Charter: Turbulence Risk Prediction

## 🎯 Scope
Develop a machine learning system to predict short‑term turbulence risk along aircraft flight paths using open weather and flight telemetry data.

## ⏱️ Prediction Horizon
Forecast turbulence risk within **0–30 minutes ahead** of the current flight segment.

## 📊 Labels
 **Binary classification**:  
  - `1` = turbulence (moderate/severe PIREP)  
  - `0` = no turbulence / light turbulence

## 📂 Data Sources
- NOAA METAR/TAF reports (weather observations/forecasts)  
- ERA5 reanalysis (atmospheric profiles)  
- OpenSky Network ADS‑B telemetry (flight tracks)  
- PIREPs (pilot turbulence reports)

## ⚙️ Models
- Random Forest classifier (tabular weather features)  
- LSTM network (sequential flight telemetry)  
- Ensemble framework combining both outputs

## 📈 Evaluation Metrics
- Accuracy, Precision, Recall, ROC‑AUC  
- Probability calibration (Brier score)  
- SHAP values for interpretability

## 🚀 Deliverables
- ETL scripts for historical + API ingestion  
- Clean dataset in Parquet format  
- Trained Random Forest + optional LSTM models  
- Ensemble inference pipeline  
- Streamlit dashboard demo  
- Final report + presentation slides

## ⚠️ Risks
- Sparse turbulence labels (PIREPs coverage limited)  
- API rate limits (OpenSky, NOAA)  
- Class imbalance (few turbulence events vs many non‑events)  
- Real‑time latency constraints

## ✅ Success Criteria
- ROC‑AUC ≥ 0.75 on held‑out test data  
- One‑command demo reproducibility (`make demo`)  
- Streamlit dashboard with interpretable predictions
