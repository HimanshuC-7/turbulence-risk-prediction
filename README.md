# Turbulence Risk Prediction for Flight Segments

## ✈️ Overview
Atmospheric turbulence is a major challenge for aviation safety, passenger comfort, and operational efficiency.  
This project builds a **machine learning system** that predicts short‑term turbulence risk along aircraft flight paths using **open weather and flight telemetry data**.

The system integrates:
- **Meteorological inputs** (wind speed, temperature, pressure, altitude)
- **Flight telemetry** (speed, climb rate, heading)
- **Labels** from PIREPs (pilot turbulence reports)

Two complementary models are trained:
- **Random Forest classifier** for tabular weather features
- **LSTM network** for sequential flight telemetry

An **ensemble framework** combines both to output a binary turbulence prediction (yes/no) with a confidence score.  
Predictions are visualized in a **Streamlit dashboard** with risk levels, feature importance, and model confidence.

---

## 📂 Project Structure
```
turbulence-risk/
├── README.md
├── project_charter.md
├── environment.yml
├── .gitignore
├── data/
│   ├── sample/
│   │   ├── sample_adsb.csv
│   │   ├── sample_metar.csv
│   │   └── sample_pireps.csv
│   └── processed/
├── src/
│   ├── etl/
│   │   └── smoke_etl.py
│   ├── features/
│   │   └── feature_engineering.py
│   └── models/
│       └── train_rf.py
├── deploy/
│   └── app.py
└── tests/
    └── test_etl.py
```
---
## 🔒 License
All Rights Reserved.  
This project is proprietary and confidential.  
No permission is granted to copy, modify, or distribute any part of this project.

---

## 👨‍💻 Author
**Project owner:** Himanshu  
**Location:** Telangana, India

