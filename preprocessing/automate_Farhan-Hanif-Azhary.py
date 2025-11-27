import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# --- KONFIGURASI PATH---
INPUT_PATH = '/home/hanif/Eksperimen_SML_Farhan-Hanif-Azhary/heart_failure_clinical_records_dataset-raw.csv'
OUTPUT_PATH = '/home/hanif/Eksperimen_SML_Farhan-Hanif-Azhary/preprocessing/heart_failure_clinical_records_dataset-clean.csv'

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan di: {path}")
    return pd.read_csv(path)

def preprocess_data(df):
    numerical_cols = [
        'age', 'creatinine_phosphokinase', 'ejection_fraction', 
        'platelets', 'serum_creatinine', 'serum_sodium', 'time'
    ]
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def save_data(df, path):
    df.to_csv(path, index=False)

if __name__ == "__main__":
    try:
        # Eksekusi Pipeline
        df = load_data(INPUT_PATH)
        df_clean = preprocess_data(df)
        save_data(df_clean, OUTPUT_PATH)
        print(f"Data berhasil diproses dan disimpan di: {OUTPUT_PATH}")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")