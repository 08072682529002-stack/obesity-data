#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import streamlit as st
import numpy as np
import pandas as pd


# In[19]:


st.set_page_config(page_title="Prediksi Tingkat Obesitas", page_icon="🧬", layout="wide")


# In[21]:


st.markdown("""
    <style>
        body {
            background-color: #fef6fb;
            font-family: 'Poppins', sans-serif;
        }
        .header-container {
            background: linear-gradient(90deg, #f9a8d4 0%, #c084fc 100%);
            color: white;
            padding: 30px 0;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 30px;
        }
        .header-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 5px;
        }
        .header-subtitle {
            font-size: 18px;
            font-weight: 400;
        }
        .stButton>button {
            background: linear-gradient(90deg, #ec4899, #a855f7);
            color: white;
            border: none;
            border-radius: 20px;
            padding: 0.7em 2em;
            font-size: 17px;
            font-weight: 600;
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #a855f7, #ec4899);
        }
        .footer {
            text-align: center;
            color: #888;
            margin-top: 50px;
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)


# In[23]:


st.markdown("""
    <div class="header-container">
        <div class="header-title">Prediksi Tingkat Obesitas</div>
        <div class="header-subtitle">Prediksi Tingkat Obesitas Menggunakan Algoritma Naive Bayes</div>
    </div>
""", unsafe_allow_html=True)


# In[25]:


with open("obesity_model.sav", "rb") as file:
    data = pickle.load(file)
model = data["model"]
scaler = data["scaler"]
columns = data["columns"]


# In[27]:


st.markdown("### 🔢 Masukkan Data Pasien di Bawah Ini")
col1, col2 = st.columns(2)
inputs = {}

for i, col in enumerate(columns):
    with (col1 if i % 2 == 0 else col2):
        inputs[col] = st.number_input(
            f"{col}",
            value=0.0,
            format="%.2f",
            help=f"Masukkan nilai untuk {col}"
        )


# In[29]:


if st.button("🔍 Prediksi Sekarang"):
    X_input = pd.DataFrame([inputs])
    X_scaled = scaler.transform(X_input)
    pred = model.predict(X_scaled)[0]

    st.markdown("---")
    st.success(f"🎯 **Hasil Prediksi Tingkat Obesitas:** {pred}")
    st.markdown("---")

    st.balloons()


# In[31]:


st.markdown("""
---
<p style="text-align:center; color:#888;">
Dibuat oleh <b>Rahma Yuliana</b> ❤️ | Naive Bayes Model Deployment
</p>
""", unsafe_allow_html=True)

