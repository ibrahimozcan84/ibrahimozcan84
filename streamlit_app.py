import streamlit as st
import joblib
import numpy as np
from scipy.optimize import differential_evolution

# Sayfa Yapılandırması
st.set_page_config(page_title="Concrete Recipe Optimizer (Beton Reçete Optimize Edici)", layout="centered")
st.title("🏗️ Concrete Mix Design System (Beton Karışım Tasarımı Sistemi)")
st.write("Enter your target strength, and let the AI ​​calculate the ideal formula. (Hedeflediğiniz dayanımı girin, yapay zeka en ideal reçeteyi hesaplasın.)")


# 1. Modeli Yükle
@st.cache_resource  # Modelin her seferinde yüklenip sistemi yavaşlatmasını engeller
def load_model():
    return joblib.load('xgboost_concrete_model.pkl')


model = load_model()

# 2. Kullanıcı Giriş Paneli (Sidebar)
with st.sidebar:
    st.header("Target Parameters   (Hedef Parametreler)")
    target_s = st.number_input("Target Strength   (Hedef Dayanım) (MPa)", min_value=10.0, max_value=85.0, value=40.0)
    target_a = st.number_input("Cure Age (Day)   Kür Yaşı (Gün)", min_value=1, max_value=120, value=28)
    st.divider()
    st.write("Optimization determines the sensitivity. (Optimizasyon hassasiyetini belirler.)")
    run_btn = st.button("Generate a Prescription(Reçete Üret)", type="primary")


# 3. Optimizasyon Fonksiyonu
def optimize_mix(trial_mix):
    wc_ratio = trial_mix[3] / trial_mix[0]
    full_input = np.append(trial_mix, wc_ratio).reshape(1, -1)
    pred = model.predict(full_input)[0]
    return abs(pred - target_s)


# 4. Ana Ekran ve Sonuçlar
if run_btn:
    with st.spinner("Artificial intelligence tries thousands of combinations...    (Yapay zeka binlerce kombinasyonu deniyor...)"):
        bounds = [(100, 500), (0, 350), (0, 200), (120, 230), (0, 30), (700, 1100), (600, 950), (target_a, target_a)]
        result = differential_evolution(optimize_mix, bounds, tol=0.01)

        optimized_params = result.x
        wc_final = optimized_params[3] / optimized_params[0]
        final_pred = model.predict(np.append(optimized_params, wc_final).reshape(1, -1))[0]

    # Görsel Sonuç Paneli
    st.success(f"Calculation Complete! Estimated Strength:   (Hesaplama Tamamlandı! Tahmini Dayanım:) {final_pred:.2f} MPa")

    col1, col2 = st.columns(2)
    features = ["Cement(Çimento)", "Slag(Cüruf)", "Fly Ash(Uçucu Kül)", "Water(Su)", "Superplasticizer(Süperakışkanlaştırıcı)", "Coarse Aggregate(Kaba Agrega)", "Fine Aggregate(İnce Agrega)"]

    with col1:
        st.metric("Target Deviation   (Hedef Sapması)", f"{abs(final_pred - target_s):.4f} MPa")
        for i in range(4):
            st.write(f"**{features[i]}:** {optimized_params[i]:.2f} kg/m³")

    with col2:
        st.metric("Water/Cement Ratio   (Su/Çimento Oranı)", f"{wc_final:.3f}")
        for i in range(4, 7):
            st.write(f"**{features[i]}:** {optimized_params[i]:.2f} kg/m³")
