import cv2
import os
import sys

# Pastikan folder dataset ada
if not os.path.exists('dataset'):
    os.makedirs('dataset')

cam = cv2.VideoCapture(0)
cam.set(3, 640) # Lebar
cam.set(4, 480) # Tinggi

# Fix for cv2.data attribute error
if hasattr(cv2, 'data'):
    FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
else:
    # Fallback 1: Inside cv2 package
    FACE_CASCADE_PATH = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascades', 'haarcascade_frontalface_default.xml')
    
    # Fallback 2: Conda Windows path
    if not os.path.exists(FACE_CASCADE_PATH):
        FACE_CASCADE_PATH = os.path.join(sys.prefix, 'Library', 'etc', 'haarcascades', 'haarcascade_frontalface_default.xml')

face_detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Masukkan ID (Ketik 1 untuk kamu)
face_id = input('\n Masukkan ID User (Tekan 1 lalu Enter): ')

print("\n [INFO] Menatap kamera... Tunggu proses pengambilan gambar...")
count = 0

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        
        # Simpan foto ke folder dataset
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        print(f"Ambil Data: {count}/30")

        cv2.imshow('Ambil Dataset', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27: # Tekan ESC buat batal
        break
    elif count >= 30: # Berhenti setelah 30 foto
        break

print("\n [INFO] Selesai. Tutup program.")
cam.release()
cv2.destroyAllWindows()