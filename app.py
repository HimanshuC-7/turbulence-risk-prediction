from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "random_forest_turbulence_model.joblib")

model = joblib.load(MODEL_PATH)

airport_map = {"VABB": 0, "VIDP": 1, "VOHS": 2}
airport_data = {
    "VABB": {"lat": 19.088686, "lon": 72.867919, "elev": 11},
    "VIDP": {"lat": 28.584511, "lon": 77.205783, "elev": 215},
    "VOHS": {"lat": 17.2333333, "lon": 78.4166666, "elev": 617},
}

skyc1_map = {"BKN": 0, "CLR": 1, "FEW": 2, "NSC": 3, "SCT": 4, "VV": 5}
skyc2_map = {"BKN": 0, "CLR": 1, "FEW": 2, "OVC": 3, "SCT": 4}
skyc3_map = {"BKN": 0, "CLR": 1, "FEW": 2, "NSC": 3, "OVC": 4, "SCT": 5}
skyc4_map = {"BKN": 0, "CLR": 1, "FEW": 2, "OVC": 3, "SCT": 4}

all_wx_cols = [
    'wx_+SHRA','wx_+TSRA','wx_-SHRA','wx_-TSRA','wx_-TSRA,fog','wx_-TSRA,haze',
    'wx_BLDU','wx_Heavy DZ','wx_Heavy RA','wx_In the vicinity RA','wx_In the vicinity SH',
    'wx_In the vicinity TS','wx_Light DZ','wx_Light RA','wx_Light TS','wx_MIFG',
    'wx_MIFG,fog','wx_Moderate BR','wx_Moderate DS','wx_Moderate DU','wx_Moderate DZ',
    'wx_Moderate FG','wx_Moderate FU','wx_Moderate HZ','wx_Moderate RA','wx_Moderate TS',
    'wx_NoWX','wx_NoWX,fog','wx_NoWX,gusty','wx_NoWX,haze','wx_SHRA',
    'wx_SHRA,haze,tempo,gusty','wx_TSRA'
]

def standardize_wxcode(code):
    if pd.isna(code) or code.lower() in ["nowx", ""]:
        return "wx_NoWX"
    codes = re.split('[, ]+', code)
    for c in codes:
        if not c:
            continue
        intensity = "Moderate"
        if c.startswith('-'):
            intensity = "Light"
            c = c[1:]
        elif c.startswith('+'):
            intensity = "Heavy"
            c = c[1:]
        elif c.startswith('VC'):
            intensity = "In the vicinity"
            c = c[2:]
        valid_codes = ['DZ','RA','SN','SG','IC','PL','GR','GS','UP',
                       'BR','FG','FU','VA','DU','SA','HZ','PY','PO',
                       'SQ','FC','SS','DS','TS','SH']
        if c in valid_codes:
            return f"wx_{intensity} {c}"
    return f"wx_{code}"

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        airport = request.form.get('airport')
        latitude = float(request.form.get('latitude', 0))
        longitude = float(request.form.get('longitude', 0))
        elevation = float(request.form.get('elevation', 0))
        tmpf_c = float(request.form.get('tmp', 0))
        dwpf_c = float(request.form.get('dwpt', 0))
        relh = float(request.form.get('relh', 0))
        drct = float(request.form.get('drct', 0))
        sknt = float(request.form.get('sknt', 0))
        gust = float(request.form.get('gust', 0))
        vsby = float(request.form.get('vsby', 0))
        mslp = float(request.form.get('mslp', 0))

        skyc1 = skyc1_map.get(request.form.get('skyc1', ''), 0)
        skyc2 = skyc2_map.get(request.form.get('skyc2', ''), 0)
        skyc3 = skyc3_map.get(request.form.get('skyc3', ''), 0)
        skyc4 = skyc4_map.get(request.form.get('skyc4', ''), 0)
        skyl1 = float(request.form.get('skyl1', 0))
        skyl2 = float(request.form.get('skyl2', 0))
        skyl3 = float(request.form.get('skyl3', 0))
        skyl4 = float(request.form.get('skyl4', 0))

        wx_input = request.form.get('wxcode', '')
        standardized_wx = standardize_wxcode(wx_input)
        wx_dict = {col:0 for col in all_wx_cols}
        if standardized_wx in wx_dict:
            wx_dict[standardized_wx] = 1

        temp_diff = tmpf_c - dwpf_c
        humidity_temp_diff = relh * temp_diff
        pressure_visibility = mslp / (vsby + 0.1)
        wind_temp = sknt * temp_diff

        has_drizzle = 1 if "DZ" in wx_input else 0
        has_rain = 1 if "RA" in wx_input else 0
        has_fog = 1 if "FG" in wx_input else 0
        has_haze = 1 if "HZ" in wx_input else 0
        has_gust = 1 if gust > 0 else 0
        has_storm = 1 if "TS" in wx_input else 0
        has_dust = 1 if "DU" in wx_input else 0

        low_visibility = 1 if vsby < 3 else 0

        input_dict = {
            'tmpf_c': tmpf_c, 'dwpf_c': dwpf_c, 'relh': relh, 'drct': drct, 'sknt': sknt,
            'mslp': mslp, 'vsby': vsby, 'gust': gust,
            'skyc1': skyc1, 'skyc2': skyc2, 'skyc3': skyc3, 'skyc4': skyc4,
            'skyl1': skyl1, 'skyl2': skyl2, 'skyl3': skyl3, 'skyl4': skyl4,
            'airport': airport_map.get(airport, 0),
            'hour': 12, 'low_visibility': low_visibility,
            'LATITUDE': latitude, 'LONGITUDE': longitude, 'ELEVATION': elevation,
            'hour_of_day': 12, 'day': 1, 'month': 1,
            'temp_diff': temp_diff, 'humidity_temp_diff': humidity_temp_diff,
            'pressure_visibility': pressure_visibility, 'wind_temp': wind_temp,
            'has_drizzle': has_drizzle, 'has_rain': has_rain, 'has_fog': has_fog,
            'has_haze': has_haze, 'has_gust': has_gust, 'has_storm': has_storm,
            'has_dust': has_dust
        }
        input_dict.update(wx_dict)

        input_df = pd.DataFrame([input_dict])
        input_df = input_df[model.feature_names_in_]

        # Prediction + probability
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        confidence = round(max(probability) * 100, 2)

        # Styled output
        if prediction == 0:
            message = f"""
            <div class='result safe'>
                ✔ SAFE TO OPERATE<br><br>
                Confidence: {confidence}%<br>
                Turbulence Risk: LOW
            </div>
            """
        else:
            message = f"""
            <div class='result danger'>
                ⚠ TURBULENCE DETECTED<br><br>
                Confidence: {confidence}%<br>
                Risk Level: HIGH
            </div>
            """

        return render_template("index.html", prediction_text=message)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
