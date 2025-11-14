import cv2
import numpy as np
import time

# Load face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def smooth(new_box, old_box, alpha=0.2):
    if old_box is None:
        return new_box
    (x, y, w, h) = new_box
    (ox, oy, ow, oh) = old_box
    return (int(ox + alpha*(x-ox)),
            int(oy + alpha*(y-oy)),
            int(ow + alpha*(w-ow)),
            int(oh + alpha*(h-oh)))


def detect_hp(frame):   # <--- nama fungsi diganti
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV)

    # noise reduction
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        if area < 1200:
            continue

        aspect = h / float(w)

        # shape HP (persegi panjang)
        if 1.3 < aspect < 4.8:
            if area > best_area:
                best_area = area
                best = (x, y, w, h)

    return best


def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    people = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(70, 70))
    return people


cap = cv2.VideoCapture(0)

prev_hp = None
prev_faces = []
fps_prev = time.time()

lost_hp = 0
MAX_LOST = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS
    now = time.time()
    fps = 1 / (now - fps_prev)
    fps_prev = now

    # ===========================
    # HP TRACKING
    # ===========================
    hp = detect_hp(frame)

    if hp is not None:
        prev_hp = smooth(hp, prev_hp)
        lost_hp = 0
    else:
        lost_hp += 1
        if lost_hp > MAX_LOST:
            prev_hp = None

    # Draw HP box
    if prev_hp is not None:
        (x, y, w, h) = prev_hp
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(frame, "HP", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ===========================
    # WAJAH TRACKING
    # ===========================
    faces = detect_faces(frame)
    new_faces = []

    for (fx, fy, fw, fh) in faces:
        new_faces.append((fx, fy, fw, fh))

    # Smoothing wajah ringan
    if prev_faces:
        smoothed = []
        for i, face in enumerate(new_faces):
            if i < len(prev_faces):
                smoothed.append(smooth(face, prev_faces[i], alpha=0.15))
            else:
                smoothed.append(face)
        prev_faces = smoothed
    else:
        prev_faces = new_faces

    # Draw face box
    for (fx, fy, fw, fh) in prev_faces:
        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 3)
        cv2.putText(frame, "Face", (fx, fy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # ===========================
    # FPS Display
    # ===========================
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("HP + Face Detector (Stable)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
