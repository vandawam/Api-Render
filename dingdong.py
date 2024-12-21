import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Data yang diberikan
data = [
    {
        "bets": {
            "Apel": 0,
            "Jeruk": 1,
            "Kelapa": 1,
            "Lonceng": 0,
            "Semangka": 0,
            "Bintang": 0,
            "Sembilan": 0,
            "Yes": 0
        },
        "out": "Jeruk1"
    },
    {
        "bets": {
            "Apel": 0,
            "Jeruk": 0,
            "Kelapa": 0,
            "Lonceng": 0,
            "Semangka": 1,
            "Bintang": 1,
            "Sembilan": 0,
            "Yes": 0
        },
        "out": "Semangka"
    },
    {
        "bets": {
            "Apel": 0,
            "Jeruk": 0,
            "Kelapa": 1,
            "Lonceng": 1,
            "Semangka": 0,
            "Bintang": 0,
            "Sembilan": 0,
            "Yes": 0
        },
        "out": "Yes"
    },
    {
        "bets": {
            "Apel": 0,
            "Jeruk": 1,
            "Kelapa": 0,
            "Lonceng": 0,
            "Semangka": 0,
            "Bintang": 0,
            "Sembilan": 0,
            "Yes": 0
        },
        "out": "Apel"
    },
    {
        "bets": {
            "Apel": 0,
            "Jeruk": 0,
            "Kelapa": 0,
            "Lonceng": 0,
            "Semangka": 0,
            "Bintang": 1,
            "Sembilan": 1,
            "Yes": 0
        },
        "out": "Sembilan1"
    },
    {
        "bets": {
            "Apel": 0,
            "Jeruk": 1,
            "Kelapa": 1,
            "Lonceng": 0,
            "Semangka": 0,
            "Bintang": 0,
            "Sembilan": 0,
            "Yes": 0
        },
        "out": "Lonceng"
    },
    {
        "bets": {
            "Apel": 1,
            "Jeruk": 0,
            "Kelapa": 0,
            "Lonceng": 1,
            "Semangka": 0,
            "Bintang": 0,
            "Sembilan": 0,
            "Yes": 0
        },
        "out": "Semangka1"
    }
]

# Preprocessing data
rows = []
for entry in data:
    row = entry["bets"]
    row["out"] = entry["out"]
    rows.append(row)

# Membuat DataFrame
df = pd.DataFrame(rows)
df.fillna(0, inplace=True)  # Mengatasi nama yang salah e.g., "Semangka ", "lonceng"

# Memisahkan fitur dan target
X = df.drop("out", axis=1)
y = df["out"]

# Encoding target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Membuat model
model = RandomForestClassifier()
model.fit(X, y_encoded)

# Fungsi prediksi
def predict_next_probabilities(input_bets):
    input_df = pd.DataFrame([input_bets])
    probabilities = model.predict_proba(input_df)[0]
    classes = label_encoder.inverse_transform(range(len(probabilities)))
    
    # Membuat hasil dalam format persentase
    result = {classes[i]: round(prob * 100, 2) for i, prob in enumerate(probabilities)}
    return result

# Contoh prediksi baru
new_data = {
    "Apel": 1,
    "Jeruk": 0,
    "Kelapa": 0,
    "Lonceng": 1,
    "Semangka": 0,
    "Bintang": 0,
    "Sembilan": 0,
    "Yes": 0
}

prediction_probabilities = predict_next_probabilities(new_data)

# Menampilkan hasil prediksi
for key, value in prediction_probabilities.items():
    print(f"{key}: {value:.2f}%")
