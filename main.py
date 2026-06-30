import cv2
import mediapipe as mp
import math
import pygame
import numpy as np
import time # <-- Tambahan krusial untuk fitur Timer 5 Detik

# ==========================================
# 1. KUMPULAN FUNGSI 
# ==========================================

def inisialisasi_suara():
    """Fungsi untuk menyiapkan sistem audio"""
    pygame.mixer.init()
    try:
        s_tidur = pygame.mixer.Sound("alarm.mp3") 
        s_nunduk = pygame.mixer.Sound("menghadap.mp3")
        return s_tidur, s_nunduk
    except:
        print("ERROR: Pastikan file alarm.mp3 dan menghadap.mp3 ada di folder!")
        exit()

def inisialisasi_model():
    """Memuat model AI beserta alat penggambar kerangka (Bones)"""
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    return face_mesh, mp_drawing, mp_drawing_styles, mp_face_mesh

def hitung_jarak(titik1, titik2):
    return math.hypot(titik2.x - titik1.x, titik2.y - titik1.y)

def hitung_ear(landmarks):
    atas = landmarks[159]
    bawah = landmarks[145]
    kiri = landmarks[33]
    kanan = landmarks[133]
    j_vertikal = hitung_jarak(atas, bawah)
    j_horizontal = hitung_jarak(kiri, kanan)
    return j_vertikal / j_horizontal if j_horizontal != 0 else 0

def cek_noleh(landmarks):
    hidung = landmarks[1]
    p_kiri = landmarks[234]
    p_kanan = landmarks[454]
    j_kiri = hitung_jarak(hidung, p_kiri)
    j_kanan = hitung_jarak(hidung, p_kanan)
    if j_kanan == 0: return True
    rasio = j_kiri / j_kanan
    return True if rasio < 0.6 or rasio > 1.6 else False

def efek_cipratan_air(frame):
    """Fungsi animasi visual disiram air"""
    tinggi, lebar = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (lebar, tinggi), (255, 150, 50), -1) 
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    jumlah_tetesan = np.random.randint(80, 150) 
    for _ in range(jumlah_tetesan):
        x = np.random.randint(0, lebar)
        y = np.random.randint(0, tinggi)
        r = np.random.randint(5, 40) 
        warna_air = (255, np.random.randint(150, 220), np.random.randint(0, 100))
        cv2.circle(frame, (x, y), r, warna_air, -1)
        cv2.circle(frame, (x - r//4, y - r//4), max(1, r//4), (255, 255, 255), -1)
    return frame


# ==========================================
# 2. PROGRAM UTAMA 
# ==========================================
def jalankan_sistem_deteksi():
    suara_tidur, suara_nunduk = inisialisasi_suara()
    face_mesh, mp_drawing, mp_drawing_styles, mp_face_mesh = inisialisasi_model()
    
    cap = cv2.VideoCapture(0) # Ganti URL IP Webcam HP di sini kalau pakai HP
    
    # Variabel Timer dan Logika Noleh
    waktu_mulai_merem = None 
    frame_wajah_hilang = 0 

    print("Sistem Berjalan... Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tinggi, lebar, _ = frame.shape 
        
        status_ngantuk = False
        status_nunduk = False

        # --- FASE 1: PROSES AI & MENGGAMBAR UI MODERN (PUTUS-PUTUS) ---
        hasil = face_mesh.process(rgb_frame)
        
        if hasil.multi_face_landmarks:
            for face_landmarks in hasil.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # ----------------------------------------------------
                # DESAIN UI FUTURISTIK: Efek Garis Putus-Putus (Dotted)
                # ----------------------------------------------------
                idx_mata_kiri = set([p for conn in mp_face_mesh.FACEMESH_LEFT_EYE for p in conn])
                idx_mata_kanan = set([p for conn in mp_face_mesh.FACEMESH_RIGHT_EYE for p in conn])
                idx_alis_kiri = set([p for conn in mp_face_mesh.FACEMESH_LEFT_EYEBROW for p in conn])
                idx_alis_kanan = set([p for conn in mp_face_mesh.FACEMESH_RIGHT_EYEBROW for p in conn])
                
                titik_ui_modern = idx_mata_kiri | idx_mata_kanan | idx_alis_kiri | idx_alis_kanan
                
                for idx in titik_ui_modern:
                    posisi = landmarks[idx]
                    x = int(posisi.x * lebar)
                    y = int(posisi.y * tinggi)
                    cv2.circle(frame, (x, y), 1, (255, 255, 0), -1) 
                    cv2.circle(frame, (x, y), 3, (150, 150, 0), 1) 
                
                # ----------------------------------------------------
                # LOGIKA NGANTUK (TIMER 5 DETIK) & NOLEH
                # ----------------------------------------------------
                if cek_noleh(landmarks):
                    frame_wajah_hilang += 1
                    waktu_mulai_merem = None # Batal ngantuk kalau noleh
                else:
                    frame_wajah_hilang = 0 
                    ear = hitung_ear(landmarks)
                    
                    if ear < 0.22: 
                        # Jika baru merem, mulai stopwatch!
                        if waktu_mulai_merem is None:
                            waktu_mulai_merem = time.time()
                        else:
                            # Hitung durasi merem
                            durasi = time.time() - waktu_mulai_merem
                            # Jika sudah 5 detik atau lebih
                            if durasi >= 5.0:
                                status_ngantuk = True
                    else: 
                        # Mata melek? Reset stopwatch ke nol!
                        waktu_mulai_merem = None 

            # Cek Trigger Eksekusi Nunduk/Noleh
            if frame_wajah_hilang > 20:
                status_nunduk = True
                
        else:
            frame_wajah_hilang += 1
            waktu_mulai_merem = None # Reset timer ngantuk kalau wajah tidak ada
            if frame_wajah_hilang > 20:
                status_nunduk = True

       # --- FASE 2: EKSEKUSI ANIMASI DAN SUARA ---
        is_audio_playing = pygame.mixer.get_busy()

        if status_ngantuk:
            frame = efek_cipratan_air(frame)
            if not is_audio_playing: suara_tidur.play() 
            cv2.rectangle(frame, (0, 0), (lebar, 80), (0, 0, 255), -1) 
            cv2.putText(frame, "!!! AWAS TIDUR !!!", (lebar//2 - 180, 55), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

        elif status_nunduk:
            if not is_audio_playing: suara_nunduk.play()
            cv2.rectangle(frame, (0, 0), (lebar, 80), (0, 140, 255), -1) 
            cv2.putText(frame, "HADAP KAMERA", (lebar//2 - 150, 55), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
            
        else:
            # JIKA KONDISI AMAN (Mata Melek & Menghadap Kamera)
            if is_audio_playing:
                pygame.mixer.stop() # Matikan semua suara seketika!

        cv2.imshow("Sistem Keselamatan Pengemudi", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()


# ==========================================
# 3. TITIK MULAI PROGRAM
# ==========================================
if __name__ == "__main__":
    jalankan_sistem_deteksi()