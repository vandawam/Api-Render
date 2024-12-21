import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer

# Data yang diberikan
data = [
    {
        "Bet": {"A": 1, "B": 1, "C": 0},
        "jackpot": False,
        "scetter": False,
        "Out": "A",
        "Pay": 20
    },
    {
        "Bet": {"A": 1, "B": 0, "C": 1},
        "jackpot": False,
        "scetter": False,
        "Out": "B",
        "Pay": 0
    },
    {
        "Bet": {"A": 1, "B": 0, "C": 1},
        "jackpot": False,
        "scetter": False,
        "Out": "B",
        "Pay": 0
    },
    {
        "Bet": {"A": 1, "B": 0, "C": 1},
        "jackpot": False,
        "scetter": True,
        "Out": ["A", "B"],
        "Pay": 20
    },
    {
        "Bet": {"A": 1, "B": 0, "C": 1},
        "jackpot": True,
        "scetter": False,
        "Out": None,
        "Pay": 200
    },
    {
        "Bet": {"A": 1, "B": 0, "C": 1},
        "jackpot": False,
        "scetter": False,
        "Out": "B",
        "Pay": 0
    },
    {
        "Bet": {"A": 1, "B": 0, "C": 1},
        "jackpot": False,
        "scetter": False,
        "Out": "B",
        "Pay": 0
    },
    {
        "Bet": {"A": 1, "B": 0, "C": 1},
        "jackpot": False,
        "scetter": False,
        "Out": "B",
        "Pay": 0
    },
    {
        "Bet": {"A": 0, "B": 1, "C": 0},
        "jackpot": False,
        "scetter": False,
        "Out": "B",
        "Pay": 10
    },{
        "Bet": {"A": 0, "B": 1, "C": 0},
        "jackpot": False,
        "scetter": False,
        "Out": "B",
        "Pay": 10
    },{
        "Bet": {"A": 0, "B": 1, "C": 0},
        "jackpot": False,
        "scetter": False,
        "Out": "B",
        "Pay": 10
    },{
        "Bet": {"A": 1, "B": 0, "C": 1},
        "jackpot": False,
        "scetter": False,
        "Out": "B",
        "Pay": 0
    },{
        "Bet": {"A": 1, "B": 1, "C": 0},
        "jackpot": False,
        "scetter": False,
        "Out": "C",
        "Pay": 0
    },{
        "Bet": {"A": 1, "B": 0, "C": 1},
        "jackpot": False,
        "scetter": False,
        "Out": "C",
        "Pay": 5
    },
]

# Preprocessing data
rows = []
for entry in data:
    row = {
        'Bet_A': entry['Bet']['A'],
        'Bet_B': entry['Bet']['B'],
        'Bet_C': entry['Bet']['C'],
        'jackpot': int(entry['jackpot']),
        'scetter': int(entry['scetter']),
        'Pay': entry['Pay']
    }
    # Mengubah "Out" menjadi nilai numerik (label encoding sederhana)
    if entry['Out'] is None:
        row['Out'] = -1
    elif isinstance(entry['Out'], list):
        row['Out'] = len(entry['Out'])  # Panjang list sebagai fitur
    else:
        row['Out'] = ord(entry['Out']) - ord('A')  # A=0, B=1, C=2, dll.
    rows.append(row)

# Membuat DataFrame
df = pd.DataFrame(rows)

# Memisahkan fitur dan target
X = df[['Bet_A', 'Bet_B', 'Bet_C', 'jackpot', 'scetter', 'Pay']]
y = df['Out']

# Membuat model
model = RandomForestClassifier()
model.fit(X, y)

# Prediksi untuk data baru (dengan probabilitas)
def predict_next_probabilities(bet_a, bet_b, bet_c, jackpot, scetter, pay):
    input_data = np.array([[bet_a, bet_b, bet_c, int(jackpot), int(scetter), pay]])
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_

    # Menghitung persentase prediksi
    result = {}
    for i, prob in enumerate(probabilities):
        if classes[i] == -1:
            result["None"] = prob * 100
        elif classes[i] >= 0 and classes[i] < 26:
            result[chr(classes[i] + ord('A'))] = prob * 100
        else:
            result[f"List with {classes[i]} elements"] = prob * 100

    return result

# Contoh prediksi baru
new_data = {
    "Bet_A": 1,
    "Bet_B": 1,
    "Bet_C": 0,
    "jackpot": False,
    "scetter": True,
    "Pay": 30
}

prediction_probabilities = predict_next_probabilities(
    new_data['Bet_A'],
    new_data['Bet_B'],
    new_data['Bet_C'],
    new_data['jackpot'],
    new_data['scetter'],
    new_data['Pay']
)

# Menampilkan hasil prediksi
for key, value in prediction_probabilities.items():
    print(f"{key}: {value:.2f}%")
