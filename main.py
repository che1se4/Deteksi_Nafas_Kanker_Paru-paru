import os
import random  # Digunakan untuk simulasi prediksi AI sementara
import librosa 
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Tentukan folder sumber dan folder tujuan sesuai struktur kita
FOLDER_DATASET = 'dataset/'
FOLDER_OUTPUT = 'output_spektrogram/'

def analisis_ai_simulasi(jalur_audio):
    """
    Fungsi ini mensimulasikan langkah 'Analisis AI: Deteksi Pola Anomali' 
    dan percabangan pada Flowchart.
    """
    # Di masa depan, bagian ini akan digantikan oleh model TensorFlow/PyTorch (.h5)
    # Untuk sekarang, kita buat simulasi logika sesuai flowchart
    
    anomali_terdeteksi = random.choice([True, False]) # Simulasi hasil deteksi AI
    
    if anomali_terdeteksi:
        # Jika YA -> Analisis Tingkat Anomali
        tingkat_anomali = random.choice(['Ringan', 'Sedang', 'Berat'])
        
        if tingkat_anomali == 'Ringan':
            skor_risiko = "Pantau Berkala"
        else:
            skor_risiko = "Segera Periksa ke Dokter"
            
        rekomendasi = (
            "- Rujukan ke Puskesmas / Rumah Sakit terdekat.\n"
            "   - Konfirmasi lanjutan via LDCT (Low-Dose CT Scan) atau Biopsi."
        )
        return "Terdeteksi Anomali (Indikasi Wheezing/Ronchi/Stridor)", tingkat_anomali, skor_risiko, rekomendasi
    else:
        # Jika TIDAK -> Skor Risiko: Aman
        skor_risiko = "Aman / Normal"
        rekomendasi = "- Tidak perlu rujukan.\n   - Lanjut monitor kondisi kesehatan napas secara berkala."
        return "Tidak Ada Pola Anomali", "-", skor_risiko, rekomendasi

def konversi_dan_analisis(folder_sumber, folder_tujuan):
    if not os.path.exists(folder_sumber):
        print(f"[ERROR] Folder '{folder_sumber}' tidak ditemukan. Silakan buat foldernya.")
        return

    # Membaca subfolder kategori data
    for kategori in os.listdir(folder_sumber):
        jalur_kategori = os.path.join(folder_sumber, kategori)
        
        if os.path.isdir(jalur_kategori):
            target_dir = os.path.join(folder_tujuan, kategori)
            os.makedirs(target_dir, exist_ok=True)
            
            files = [f for f in os.listdir(jalur_kategori) if f.endswith('.wav')]
            
            if not files:
                print(f"\n(Belum ada file .wav di folder dataset/{kategori})")
                continue
                
            print(f"\n--- Memproses Kategori: {kategori.upper()} ---")
            
            for file_audio in files:
                jalur_audio = os.path.join(jalur_kategori, file_audio)
                print(f"\n[1] Membaca rekaman: {file_audio}")
                
                try:
                    # 1. Langkah Flowchart: Konversi ke Spektrogram
                    y, sr = librosa.load(jalur_audio, sr=None)
                    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
                    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
                    
                    nama_file_gambar = file_audio.replace('.wav', '.png')
                    jalur_simpan = os.path.join(target_dir, nama_file_gambar)
                    
                    plt.figure(figsize=(10, 4))
                    librosa.display.specshow(mel_spect_db, sr=sr, hop_length=512)
                    plt.axis('off')
                    plt.savefig(jalur_simpan, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    print(f"    -> ✓ Konversi Spektrogram Selesai: {nama_file_gambar}")
                    
                    # 2. Langkah Flowchart: Analisis AI & Output Hasil
                    hasil_ai, tingkat, skor, rekomendasi = analisis_ai_simulasi(jalur_audio)
                    
                    print(f"[2] Hasil Analisis AI : {hasil_ai}")
                    if tingkat != "-":
                        print(f"[3] Tingkat Anomali   : {tingkat}")
                    print(f"[4] Skor Risiko       : {skor}")
                    print(f"[5] Rekomendasi Klinis:\n   {rekomendasi}")
                    print("-" * 40)
                
                except Exception as e:
                    print(f"    ✕ Gagal memproses {file_audio}: {e}")

if __name__ == "__main__":
    print("=====================================================")
    print(" SISTEM OBSERVASI NAPAS BERBASIS AI")
    print(" DETEKSI DINI KANKER PARU-PARU LEWAU AUDIO-SPEKTROGRAM")
    print("=====================================================")
    
    konversi_dan_analisis(FOLDER_DATASET, FOLDER_OUTPUT)
    
    print("\n================== PROSES SELESAI ==================")
