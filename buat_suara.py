from gtts import gTTS

print("Sedang membuat file suara, tunggu sebentar...")

# 1. Membuat suara untuk alarm ngantuk
teks_ngantuk = "Bayu, awas ngantuk! Bangun!"
suara1 = gTTS(text=teks_ngantuk, lang='id', slow=False)
suara1.save("alarm.mp3")

# 2. Membuat suara untuk peringatan nunduk
teks_nunduk = "Tolong menghadap kamera ya."
suara2 = gTTS(text=teks_nunduk, lang='id', slow=False)
suara2.save("menghadap.mp3")

print("Berhasil! File alarm.mp3 dan menghadap.mp3 sudah siap digunakan.")