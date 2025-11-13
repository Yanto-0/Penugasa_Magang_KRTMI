import cv2
import sys
import time  # untuk menghitung FPS

def main():
    try:
        # Load Haar cascade untuk deteksi wajah
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("Error: Gagal memuat classifier wajah.")
            sys.exit(1)

        # Buka kamera default
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Mencoba kamera indeks 1...")
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                print("Error: Tidak dapat mengakses kamera.")
                sys.exit(1)

        print("Kamera berhasil dibuka. Tekan 'q' untuk keluar.")

        prev_time = 0  # waktu frame sebelumnya

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Gagal membaca frame dari kamera.")
                break

            # Hitung FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time

            # Konversi ke grayscale untuk deteksi wajah
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Deteksi wajah
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Gambar rectangle dan label "Wajah"
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, 'Wajah', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Tampilkan FPS di pojok kiri atas
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # Tampilkan hasil
            cv2.imshow('Deteksi Wajah OpenCV - Tekan q untuk keluar', frame)

            # Tekan 'q' untuk keluar
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Bersihkan
        cap.release()
        cv2.destroyAllWindows()
        print("Program selesai.")

    except Exception as e:
        print(f"Error tak terduga: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
