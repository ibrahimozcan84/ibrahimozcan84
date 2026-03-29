import streamlit as st
import joblib
import numpy as np
import time
from scipy.optimize import differential_evolution

# Sayfa Yapılandırması
st.set_page_config(page_title="Concrete Recipe Optimizer", layout="centered")
st.title("🏗️ Concrete Mix Design System")
st.write("Enter your target strength, and let the AI calculate the ideal formula. (Hedeflediğiniz dayanımı girin, yapay zeka en ideal reçeteyi hesaplasın.)")

# 1. Modeli Yükle
@st.cache_resource
def load_model():
    return joblib.load('xgboost_concrete_model.pkl')

model = load_model()

# 2. Kullanıcı Giriş Paneli (Sidebar)
with st.sidebar:
    st.header("Target Parameters (Hedef Parametreler)")
    target_s = st.number_input("Target Strength (Hedef Dayanım) (MPa)", min_value=10.0, max_value=85.0, value=40.0)
    target_a = st.number_input("Cure Age (Day) (Kür Yaşı) (Gün)", min_value=1, max_value=120, value=28)
    st.divider()
    # Hız Ayarı: Makale için önemli bir metrik
    opt_speed = st.select_slider("Optimization Speed (Optimizasyon Hızı)", options=["Fast (Hızlı)", "Balanced (Dengeli)", "High Precision (Yüksek Hassasiyet)"], value="Balanced (Dengeli)")
    run_btn = st.button("Generate a Prescription (Reçete Üret)", type="primary")

# 3. Optimizasyon Fonksiyonu
def optimize_mix(trial_mix, target_s, target_a):
    cement = trial_mix[0]
    water = trial_mix[3]
    wc_ratio = water / cement if cement > 0 else 1.0
    
    # Mühendislik Kısıtı: Su/Çimento oranı cezalandırması (Penalization)
    penalty = 0
    if wc_ratio < 0.30 or wc_ratio > 0.75:
        penalty = 5000 

    full_input = np.append(trial_mix, wc_ratio).reshape(1, -1)
    pred = model.predict(full_input)[0]
    return abs(pred - target_s) + penalty

# 4. Takip Sistemi (Callback)
def update_ui(xk, convergence):
    percent = min(int(convergence * 100), 100)
    st.session_state.p_bar.progress(percent)
    st.session_state.s_text.text(f"Progress (İlerleme): %{percent}")

# 5. Ana Ekran ve Sonuçlar
if run_btn:
    # Arayüz elemanlarını hazırla
    st.session_state.p_bar = st.progress(0)
    st.session_state.s_text = st.empty()
    
    # Parametre Haritası
    speed_cfg = {"Fast (Hızlı)": (5, 30), "Balanced (Dengeli)": (10, 50), "High Precision (Yüksek Hassasiyet)": (15, 100)}
    pop, max_it = speed_cfg[opt_speed]

    with st.spinner("AI is calculating combinations... (Yapay zeka kombinasyonları hesaplıyor...)"):
        start_time = time.time()
        bounds = [(100, 500), (0, 350), (0, 200), (120, 230), (0, 30), (700, 1100), (600, 950), (target_a, target_a)]
        
        result = differential_evolution(
            optimize_mix, 
            bounds, 
            args=(target_s, target_a),
            callback=update_ui,
            popsize=pop,
            maxiter=max_it,
            tol=0.01
        )

        optimized_params = result.x
        wc_final = optimized_params[3] / optimized_params[0]
        final_pred = model.predict(np.append(optimized_params, wc_final).reshape(1, -1))[0]
        exec_time = time.time() - start_time

    # Görsel Sonuç Paneli
    st.session_state.p_bar.empty()
    st.session_state.s_text.empty()
    st.success(f"Calculation Complete in {exec_time:.2f}s! (Hesaplama {exec_time:.2f} saniyede tamamlandı!)")

    col1, col2 = st.columns(2)
    features = ["Cement(Çimento)", "Slag(Cüruf)", "Fly Ash(Uçucu Kül)", "Water(Su)", "Superplasticizer(Süperakışkanlaştırıcı)", "Coarse Aggregate(Kaba Agrega)", "Fine Aggregate(İnce Agrega)"]

    with col1:
        st.metric("Target Deviation (Hedef Sapması)", f"{abs(final_pred - target_s):.4f} MPa")
        for i in range(4):
            st.write(f"**{features[i]}:** {optimized_params[i]:.2f} kg/m³")

    with col2:
        st.metric("Water/Cement Ratio (Su/Çimento Oranı)", f"{wc_final:.3f}")
        for i in range(4, 7):
            st.write(f"**{features[i]}:** {optimized_params[i]:.2f} kg/m³")
