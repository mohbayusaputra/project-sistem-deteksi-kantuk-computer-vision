import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify
import cv2
import math
import numpy as np
import time
import base64

mp_face_mesh = mp.solutions.face_mesh
app = Flask(__name__)

print("[INFO] Memuat Model AI Hybrid...")
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, 
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
cnn_model = load_model("mobilenetv2_eye_classifier.keras")

# Variabel Global untuk tracking waktu antar-request
waktu_mulai_merem = None
frame_wajah_hilang = 0

def hitung_jarak(titik1, titik2):
    return math.hypot(titik2.x - titik1.x, titik2.y - titik1.y)

def cek_noleh(landmarks):
    hidung = landmarks[1]
    p_kiri = landmarks[234]
    p_kanan = landmarks[454]
    if hitung_jarak(hidung, p_kanan) == 0: return True
    rasio = hitung_jarak(hidung, p_kiri) / hitung_jarak(hidung, p_kanan)
    return True if rasio < 0.6 or rasio > 1.6 else False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global waktu_mulai_merem, frame_wajah_hilang
    
    # 1. Terima gambar Base64 dari Javascript
    data = request.json.get('image')
    if not data:
        return jsonify({"error": "No image data"}), 400
        
    # 2. Decode Base64 ke format OpenCV (numpy array)
    encoded_data = data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tinggi, lebar, _ = frame.shape
    
    lokal_ngantuk, lokal_nunduk = False, False
    
    # 3. Proses Mediapipe
    hasil = face_mesh.process(rgb_frame)
    
    if hasil.multi_face_landmarks:
        for face_landmarks in hasil.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            idx_mata_kiri = set([p for conn in mp_face_mesh.FACEMESH_LEFT_EYE for p in conn])
            
            if cek_noleh(landmarks):
                frame_wajah_hilang += 1
                waktu_mulai_merem = None 
            else:
                frame_wajah_hilang = 0 
                x_coords = [int(landmarks[i].x * lebar) for i in idx_mata_kiri]
                y_coords = [int(landmarks[i].y * tinggi) for i in idx_mata_kiri]
                pad = 10
                x_min, x_max = max(0, min(x_coords) - pad), min(lebar, max(x_coords) + pad)
                y_min, y_max = max(0, min(y_coords) - pad), min(tinggi, max(y_coords) + pad)
                
                if x_max > x_min and y_max > y_min:
                    crop_mata = rgb_frame[y_min:y_max, x_min:x_max]
                    crop_mata = cv2.resize(crop_mata, (224, 224))
                    crop_array = np.expand_dims(crop_mata, axis=0).astype('float32') / 127.5 - 1.0 
                    
                    prediksi = cnn_model(crop_array, training=False)[0][0].numpy()
                    if prediksi > 0.8: 
                        if waktu_mulai_merem is None: 
                            waktu_mulai_merem = time.time()
                        elif time.time() - waktu_mulai_merem >= 2.0: 
                            lokal_ngantuk = True
                    else:
                        waktu_mulai_merem = None 
                        
        if frame_wajah_hilang > 5: # Kurangi threshold frame hilang karena frame rate dari web bisa bervariasi
            lokal_nunduk = True
    else:
        frame_wajah_hilang += 1
        waktu_mulai_merem = None
        if frame_wajah_hilang > 5:
            lokal_nunduk = True

    return jsonify({
        "ngantuk": lokal_ngantuk,
        "nunduk": lokal_nunduk
    })

if __name__ == '__main__':
    # Di cloud, usahakan tidak pakai debug mode demi performa
    app.run(host='0.0.0.0', port=5000, debug=False)