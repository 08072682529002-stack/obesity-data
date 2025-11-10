#!/usr/bin/env python
# coding: utf-8

# In[36]:


import streamlit as st
import pickle
import pandas as pd

# =====================
# === KONFIGURASI UI ===
# =====================
st.set_page_config(page_title="Obesity Prediction App", page_icon="🩺", layout="wide")

st.markdown("""
    <style>
        body {
            background-color: #f0f8ff;
            font-family: 'Poppins', sans-serif;
        }
        .header {
            background: linear-gradient(90deg, #60a5fa, #2563eb);
            padding: 20px;
            border-radius: 12px;
            color: white;
            text-align: center;
            margin-bottom: 20px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
        }
        .header h1 {
            font-size: 42px;
            margin-bottom: 5px;
        }
        .header p {
            font-size: 18px;
            margin: 0;
        }
        .stButton>button {
            background: linear-gradient(90deg, #3b82f6, #1e40af);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.6em 2em;
            font-size: 17px;
            font-weight: 600;
            transition: 0.3s;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #1e40af, #3b82f6);
        }
        .footer {
            text-align: center;
            color: #666;
            margin-top: 40px;
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'><h1>Prediksi Tingkat Obesitas</h1><p>Prediksi Kategori Obesitas Berdasarkan Pola Hidup</p></div>", unsafe_allow_html=True)
st.write("")

# =====================
# === LOAD MODEL ===
# =====================
try:
    with open("obesity_model.sav", "rb") as file:
        saved = pickle.load(file)
except FileNotFoundError:
    st.error("❌ File 'obesity_model.sav' tidak ditemukan. Jalankan dulu training model.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()

# semua variabel model diinisialisasi di luar blok try
model = saved["model"]
scaler = saved["scaler"]
columns = saved["columns"]
label_encoders = saved["label_encoders"]
target_col = saved.get("target", "ObesityCategory")

# =====================
# === INPUT DATA ===
# =====================

st.markdown("### 🔢 Masukkan Data Pasien di Bawah Ini")

input_values = {}
col1, col2 = st.columns(2)

for i, col in enumerate(columns):
    container = col1 if i % 2 == 0 else col2

    # Jika kolom kategorikal (pakai label encoder)
    if col in label_encoders:
        le = label_encoders[col]
        classes = list(le.classes_)
        options = classes

        with container:
            choice = st.selectbox(f"{col}", options, index=0, key=f"sel_{col}")
            if choice == "Type manually":
                manual = st.text_input(f"Ketik nilai {col} (contoh: {classes[0]})", key=f"txt_{col}")
                value_to_encode = manual.strip()
            else:
                value_to_encode = choice

            # Cocokkan huruf besar/kecil
            matched = None
            for c in classes:
                if str(c).lower() == str(value_to_encode).lower():
                    matched = c
                    break

            if matched is None:
                st.warning(f"⚠️ Nilai '{value_to_encode}' untuk {col} tidak valid. Gunakan salah satu: {classes}")
                input_values[col] = None
            else:
                encoded = int(le.transform([matched])[0])
                input_values[col] = encoded

    else:
        with container:
            val = st.number_input(f"{col}", value=0.0, format="%.2f", key=f"num_{col}")
            input_values[col] = float(val)

# =====================
# === PREDIKSI ===
# =====================
if st.button("🔍 Prediksi Kategori Obesitas"):
    if None in input_values.values():
        st.error("❌ Masih ada input yang kosong atau tidak valid.")
    else:
        X_input = pd.DataFrame([[input_values[c] for c in columns]], columns=columns)
        X_scaled = scaler.transform(X_input)
        pred = model.predict(X_scaled)[0]

        # Kembalikan ke label asli kategori obesitas
        target_le = label_encoders.get(target_col)
        if target_le is not None:
            pred_label = target_le.inverse_transform([int(pred)])[0]
        else:
            pred_label = str(pred)

        st.markdown("---")
        st.markdown(
            f"<h3 style='text-align:center; color:##1e3a8a;'>➡️ Hasil Prediksi: "
            f"<span style='color:#2563eb;'> {pred_label}</span></h3>",
            unsafe_allow_html=True
        )
        st.balloons()

st.markdown("<br><p style='text-align:center;color:#888;'>Dibuat oleh <b>Rahma Yuliana 💕 </b></p>", unsafe_allow_html=True)


# In[ ]:




