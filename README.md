# ✈️ Turbulence Risk Prediction for Flight Segments

## Overview

Atmospheric turbulence is one of the major challenges in aviation, affecting **passenger comfort, flight safety, and operational efficiency**.  

This project implements a **machine learning system** to predict **short-term turbulence risk** along aircraft flight paths using **weather observations (METAR, ISD) and flight telemetry**.  

Unlike traditional turbulence forecasting methods, which rely on pilot reports (PIREPs) that are scarce in some regions, this system provides **data-driven predictions** even in areas like India where PIREPs are limited.

---

## Dataset

- **Airports Covered:**  
  - VOHS (Shirdi Airport)  
  - VIDP (Indira Gandhi International Airport, Delhi)  
  - VABB (Chhatrapati Shivaji Maharaj International Airport, Mumbai)  

- **Weather Data Sources:**  
  - METAR (routine aviation weather reports)  
  - ISD (Integrated Surface Data)  

- **Flight Telemetry Features:**  
  - Aircraft speed, heading, climb rate, and altitude  

- **Total Features:** 41 meteorological and flight-related columns, including:  
  - Temperature (`tmp`) and Dewpoint (`dwpt`)  
  - Relative Humidity (`relh`)  
  - Wind speed (`sknt`), wind direction (`drct`), gust (`gust`)  
  - Visibility (`vsby`)  
  - Cloud layers (`skyl1` – `skyl4`)  
  - Pressure (`mslp`)  
  - Other engineered features like `dewpoint_dep`, `low_visibility`, `is_day`, `num_cloud_layers`  

- **Target Variable:**  
  - `turbulence` (binary: 0 = No turbulence, 1 = Turbulence)  
  - Created based on weather thresholds rather than PIREPs, making it suitable for India.  

- **Dataset Size:** 51,266 records  

---

## Data Preprocessing

1. Cleaned and merged METAR and ISD data for all three airports.  
2. Feature engineering:  
   - Computed cloud layers, temperature differences, humidity-based features.  
   - Removed features that could leak label information: `gust_factor`, `wind_speed_diff`, `wind_dir_diff`, `tmp_diff`, `mslp_diff`, `high_wind_gust`.  
3. Sorted dataset by date for **time-series-aware validation**.  

---

## Modeling Approach

- **Model Used:** Random Forest Classifier  
- **Validation Strategy:** `TimeSeriesSplit` to respect temporal ordering (no shuffling)  
- **Metrics Evaluated:** Accuracy, Precision, Recall, F1-score, ROC-AUC  

**Random Forest Parameters:**

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)
```
## Model Performance

| Metric     | Mean Score (%) |
|-----------|----------------|
| Accuracy  | 95.6           |
| Precision | 96.97          |
| Recall    | 97.34          |
| F1-score  | 97.13          |
| ROC-AUC   | 98.74          |

## Feature Importance

The top features influencing turbulence predictions include:

- **Wind speed (`sknt`)** – mechanical turbulence likelihood
- **Visibility (`vsby`)** – low visibility correlates with unstable conditions
- **Cloud base (`skyl1`)** – low clouds can indicate convective turbulence
- **Relative humidity (`relh`)** – high humidity increases instability
- **Temperature / Dewpoint differences** – indicate thermal activity

---
## 🔒 License
All Rights Reserved.  
This project is proprietary and confidential.  
No permission is granted to copy, modify, or distribute any part of this project.

---

## 👨‍💻 Author
**Project owner:** Himanshu  
**Location:** Telangana, India

