import cv2
import numpy as np

# OpenCV'nin yüz tanıma sınıflandırıcısını yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Kamera başlatma
cap = cv2.VideoCapture(0)

# Kendi yüzünüzü eğitim verisi olarak kullanmak için resimlerinizi yükleyin
samed_image = cv2.imread("samed_bilgin.jpg", cv2.IMREAD_GRAYSCALE)  # samed.jpg dosyası kendi yüzünüzü içermelidir
samed_image = cv2.resize(samed_image, (100, 100))  # Görüntüyü istenen boyuta yeniden boyutlandırın

# Elon Musk'ın yüzünü eğitim verisi olarak kullanmak için resimlerinizi yükleyin
elon_image = cv2.imread("elon.jpg", cv2.IMREAD_GRAYSCALE)  # elon.jpg dosyası Elon Musk'ın yüzünü içermelidir
elon_image = cv2.resize(elon_image, (100, 100))  # Görüntüyü istenen boyuta yeniden boyutlandırın

# Etiketleri belirtin
labels = np.array([0, 1])  # 0: Samed Bilgin, 1: Elon Musk

# Eğitim verisini oluşturun
images = [samed_image, elon_image]  # Kendi yüzünüz ve Elon Musk'ın yüzü için eğitim verisini ekleyin

# Yüz tanıma modelini oluşturun
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images, labels)

while True:
    # Kameradan bir kare al
    ret, frame = cap.read()
    
    # Kareyi gri tona dönüştür
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Yüzleri algıla
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Algılanan yüzlerin etrafına dikdörtgen çiz
    for (x, y, w, h) in faces:
        # Algılanan yüzü eğitim verisiyle karşılaştır
        roi_gray = cv2.resize(gray[y:y+h, x:x+w], (100, 100))  # Algılanan yüz bölgesini istenen boyuta yeniden boyutlandırın
        label, confidence = recognizer.predict(roi_gray)
        if label == 0:  # Samed Bilgin
            cv2.putText(frame, 'Samed Bilgin', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        elif label == 1:  # Elon Musk
            cv2.putText(frame, 'Elon Musk', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        else:  # Tanınmayan yüz
            cv2.putText(frame, 'Unknown', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Kareyi göster
    cv2.imshow('Face Recognition', frame)
    
    # Çıkış için 'q' tuşuna basıldığında döngüyü kır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera kapatma ve pencereleri yok etme
cap.release()
cv2.destroyAllWindows()
