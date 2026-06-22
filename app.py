import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import random

# --- LOGIKA DIAGNOSIS SIMULASI ---
def analisis_ai_simulasi(jalur_audio):
    # Mengikuti alur flowchart KTI kamu kemarin
    anomali_terdeteksi = random.choice([True, False])
    if anomali_terdeteksi:
        tingkat_anomali = random.choice(['Ringan', 'Sedang', 'Berat'])
        skor_risiko = "Pantau Berkala" if tingkat_anomali == 'Ringan' else "Segera Periksa ke Dokter"
        rekomendasi = (
            "- Rujukan ke Puskesmas / Rumah Sakit terdekat.\n"
            "- Konfirmasi lanjutan via LDCT (Low-Dose CT Scan) atau Biopsi."
        )
        return "Terdeteksi Anomali (Indikasi Wheezing/Ronchi/Stridor)", tingkat_anomali, skor_risiko, rekomendasi
    else:
        return "Tidak Ada Pola Anomali", "-", "Aman / Normal", "- Tidak perlu rujukan.\n- Lanjut monitor kondisi kesehatan napas secara berkala."

# --- TAMPILAN ANTARMUKA WEB ---
st.set_page_config(page_title="Sistem Observasi Napas AI", layout="centered")

st.title("🫁 Sistem Observasi Napas Berbasis AI")
st.subheader("Deteksi Dini Anomali Paru-Paru Lewat Audio-Spektrogram")
st.write("Silakan unggah file rekaman suara napas (.wav) untuk memulai analisis otomatis.")

st.divider()

# Tombol Unggah File Audio
file_audio = st.file_uploader("Pilih file audio suara napas...", type=["wav"])

if file_audio is not None:
    st.success("✓ File audio berhasil diunggah!")
    
    # Fitur putar audio di web
    st.audio(file_audio, format="audio/wav")
    
    # 1. PROSES KONVERSI KE SPEKTROGRAM VISUAL
    st.write("### 📊 Proses Konversi Spektrogram")
    with st.spinner("Sedang mengekstrak fitur frekuensi suara..."):
        y, sr = librosa.load(file_audio, sr=None)
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(mel_spect_db, sr=sr, hop_length=512, ax=ax)
        plt.axis('off')
        
        st.pyplot(fig)
        plt.close()
    
    st.divider()
    
    # 2. PROSES ANALISIS DIAGNOSIS LOGIKA
    st.write("### 🤖 Hasil Analisis AI")
    hasil_ai, tingkat, skor, rekomendasi = analisis_ai_simulasi(file_audio)
    
    if "Terdeteksi Anomali" in hasil_ai:
        st.error(f"**Status:** {hasil_ai}")
        st.warning(f"**Tingkat Anomali:** {tingkat}")
        st.error(f"**Skor Risiko:** {skor}")
    else:
        st.success(f"**Status:** {hasil_ai}")
        st.success(f"**Skor Risiko:** {skor}")
        
    st.write("**Rekomendasi Klinis:**")
    st.info(rekomendasi)

st.divider()
st.caption("Prototipe Pengembangan Sistem KTI SMA - SMAN 3 Denpasar © 2026")