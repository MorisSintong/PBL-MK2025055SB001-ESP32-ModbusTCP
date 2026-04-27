import cv2
import numpy as np
from PIL import Image
import os
import sys

# Path ke folder dataset
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Fix for cv2.data attribute error
if hasattr(cv2, 'data'):
    FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
else:
    # Fallback 1: Inside cv2 package
    FACE_CASCADE_PATH = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascades', 'haarcascade_frontalface_default.xml')
    
    # Fallback 2: Conda Windows path
    if not os.path.exists(FACE_CASCADE_PATH):
        FACE_CASCADE_PATH = os.path.join(sys.prefix, 'Library', 'etc', 'haarcascades', 'haarcascade_frontalface_default.xml')

detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:
        # Ubah gambar ke grayscale
        PIL_img = Image.open(imagePath).convert('L') 
        img_numpy = np.array(PIL_img,'uint8')

        # Ambil ID dari nama file (User.1.jpg -> ID = 1)
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Sedang melatih data wajah... Tunggu sebentar.")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Simpan model ke file trainer.yml
if not os.path.exists('trainer'):
    os.makedirs('trainer')
    
recognizer.write('trainer/trainer.yml') 

print(f"\n [INFO] {len(np.unique(ids))} wajah telah dilatih. File trainer.yml berhasil dibuat!")