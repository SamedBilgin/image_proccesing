import cv2
import mediapipe as mp
import numpy as np

# Mediapipe pose modülü
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Web kamerasını başlat
cap = cv2.VideoCapture(0)

# Kafa sallama algılama için gerekli değişkenler
nod_counter = 0
nod_threshold = 15  # Kafa sallama eşiği (daha yüksek ayarlandı)
angle_threshold = 20  # Açı eşiği (derece)
message_displayed = False

with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Kamera erişilemedi.")
            break

        # BGR görüntüyü RGB'ye çevir
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Tespiti yap
        results = pose.process(image)

        # Tespit edilen kafa var mı kontrol et
        if results.pose_landmarks:
            # Kafa noktasını al
            head_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

            # Pixel cinsinden koordinatlara dönüştür
            head_coords = np.array([head_landmark.x * frame.shape[1], head_landmark.y * frame.shape[0]])

            # Kafa açısını hesapla
            angle = np.arctan2(head_coords[1] - frame.shape[0] / 2, head_coords[0] - frame.shape[1] / 2) * 180 / np.pi

            # Kafa sallama algıla
            if abs(angle) > angle_threshold:
                nod_counter += 1
                if nod_counter > nod_threshold:
                    cv2.putText(frame, "Kafa Sallandi mi?", (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    if not message_displayed:
                        print("Kafa Sallandı")
                        message_displayed = True
                else:
                    message_displayed = False
            else:
                nod_counter = 0
                message_displayed = False

            # Kafayı ve açıyı görselleştir
            cv2.circle(frame, (int(head_coords[0]), int(head_coords[1])), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Angle: {int(angle)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Sonucu göster
        cv2.imshow('Kafa Sallamayi Algilama', frame)

        # Çıkış için 'q' tuşuna basın
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
