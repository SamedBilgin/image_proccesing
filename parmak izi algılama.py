# -*- coding: utf-8 -*-
"""
Created on Fri May 24 00:43:58 2024

@author: samed
"""

import cv2
import numpy as np

def fingerprint_match(img1, img2):
    # İki parmak izi görüntüsünü gri tonlamalı olarak yükleyin
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # İki görüntü arasındaki benzerliği hesaplayın
    result = cv2.matchTemplate(img1_gray, img2_gray, cv2.TM_CCOEFF_NORMED)
    similarity = np.max(result)

    return similarity

# Parçalanan parmak izi görüntülerini yükleyin
fingerprint1 = cv2.imread("fingerprint1.jpg")
fingerprint2 = cv2.imread("fingerprint2.jpg")

# Parmağınızı iki farklı sensöre koyup alınan iki farklı parmak izi resmini karşılaştırın
similarity = fingerprint_match(fingerprint1, fingerprint2)

# Benzerlik değerini eşik değeriyle karşılaştırın
threshold = 0.8  # Örnek bir eşik değeri
if similarity > threshold:
    print("Parmak izi eşleşti!")
else:
    print("Parmak izi eşleşmedi.")

# Sonuçları görselleştirin
cv2.imshow("Fingerprint 1", fingerprint1)
cv2.imshow("Fingerprint 2", fingerprint2)
cv2.waitKey(0)
cv2.destroyAllWindows()
