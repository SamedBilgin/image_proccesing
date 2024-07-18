import cv2
import numpy as np

# Web kamerasını başlat
cap = cv2.VideoCapture(0)

# El algılama için Mediapipe kütüphanesini kullan
import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# El sallama kontrolü için gerekli değişkenler
saying_goodbye = False
gesture_frames = 0
gesture_threshold = 10  # El sallama eşiği
no_gesture_frames = 0
no_gesture_threshold = 50  # El sallanmadığı durumda ne kadar süre bekleyeceğimiz

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Kamera erişilemedi.")
        break

    # Görüntüyü RGB'ye dönüştür
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # El algılama işlemini gerçekleştir
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # El koordinatlarını al
        hand_landmarks = results.multi_hand_landmarks[0]

        # Baş ve işaret parmağı uç noktalarını al
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # Baş ve işaret parmağı arasındaki uzaklığı hesapla
        distance = np.sqrt((wrist.x - index_finger_tip.x) ** 2 + (wrist.y - index_finger_tip.y) ** 2)

        # El sallama algıla
        if distance > 0.1:  
            gesture_frames += 1
            no_gesture_frames = 0  # Elin sallandığı anı tespit ettiğimizde beklemeyi sıfırla
            if gesture_frames > gesture_threshold and not saying_goodbye:
                print("Güle güle!")
                saying_goodbye = True
        else:
            gesture_frames = 0
            no_gesture_frames += 1
            if no_gesture_frames > no_gesture_threshold:
                saying_goodbye = False

    # Sonucu göster
    cv2.imshow('El Sallama Algılama', frame)

    # Çıkış için 'q' tuşuna basın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
