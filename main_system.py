import cv2
import serial
import time
import mediapipe as mp
import numpy as np
import os

# -------------------------------
# CONFIG
# -------------------------------
SERIAL_PORT = 'COM8'
BAUD_RATE = 9600

# -------------------------------
# CONNECT ARDUINO
# -------------------------------
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE)
    time.sleep(2)
    print("✅ Arduino Connected")
except Exception as e:
    print("❌ Arduino connection failed:", e)
    exit()

# -------------------------------
# LOAD FACE MODEL
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "face_model.yml")
labels_path = os.path.join(BASE_DIR, "labels.npy")

if not os.path.exists(model_path):
    print("❌ Run setup_faces.py first")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

label_map = np.load(labels_path, allow_pickle=True).item()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# -------------------------------
# SENSOR CHECK
# -------------------------------
def read_sensor():

    values = {
        "vibration": None,
        "gas": None,
        "temp": None,
        "moisture": None
    }

    start = time.time()

    while time.time() - start < 5:

        if arduino.in_waiting:

            try:
                line = arduino.readline().decode().strip()

                print("📩", line)

                if ":" in line:

                    key, val = line.split(":")
                    key = key.lower()

                    if key == "vibration":
                        values[key] = val

                    else:
                        values[key] = float(val)

            except:
                pass

    print("\n📊 SENSOR VALUES")
    print(values)

    # ALERT CONDITIONS
    if (
        values["vibration"] == "HIGH" or
        (values["gas"] and values["gas"] > 100) or
        (values["temp"] and values["temp"] > 40) or
        (values["moisture"] and values["moisture"] > 2)
    ):
        return True

    return False

# -------------------------------
# FACE AUTHENTICATION
# -------------------------------
def authenticate():

    cam = cv2.VideoCapture(0)

    start = time.time()

    print("\n🔐 FACE AUTHENTICATION STARTED")

    while True:

        ret, frame = cam.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Improve low-light detection
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=(80, 80)
        )

        for (x, y, w, h) in faces:

            roi = gray[y:y+h, x:x+w]

            roi = cv2.resize(roi, (100, 100))

            try:
                label, conf = recognizer.predict(roi)

            except:
                continue

            # LOWER = BETTER
            if conf < 85:

                name = label_map[label]

                print(f"✅ AUTHENTICATED : {name}")

                cv2.putText(
                    frame,
                    f"Welcome {name}",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )

                cv2.rectangle(
                    frame,
                    (x,y),
                    (x+w,y+h),
                    (0,255,0),
                    3
                )

                cv2.imshow("Authentication", frame)

                cv2.waitKey(1500)

                cam.release()
                cv2.destroyAllWindows()

                return True

            else:

                cv2.putText(
                    frame,
                    "Unknown",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    2
                )

                cv2.rectangle(
                    frame,
                    (x,y),
                    (x+w,y+h),
                    (0,0,255),
                    3
                )

        cv2.imshow("Authentication", frame)

        # ESC KEY
        if cv2.waitKey(1) == 27:
            break

        # TIMEOUT
        if time.time() - start > 15:
            break

    cam.release()
    cv2.destroyAllWindows()

    return False

# -------------------------------
# VIRTUAL SWITCH CONTROL
# -------------------------------
def gesture_control():

    print("\n🖐 VIRTUAL SWITCH CONTROL STARTED")

    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    switch_on = False

    last_touch = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        # ---------------------------------
        # BUTTON POSITION
        # ---------------------------------

        btn_x1 = 180
        btn_y1 = 180

        btn_x2 = 470
        btn_y2 = 320

        # ---------------------------------
        # BUTTON UI
        # ---------------------------------

        if switch_on:

            color = (0,255,0)
            text = "SYSTEM ON"

        else:

            color = (0,0,255)
            text = "SYSTEM OFF"

        cv2.rectangle(
            frame,
            (btn_x1, btn_y1),
            (btn_x2, btn_y2),
            color,
            -1
        )

        cv2.rectangle(
            frame,
            (btn_x1, btn_y1),
            (btn_x2, btn_y2),
            (255,255,255),
            4
        )

        cv2.putText(
            frame,
            text,
            (btn_x1 + 30, btn_y1 + 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255,255,255),
            3
        )

        # ---------------------------------
        # HAND DETECTION
        # ---------------------------------

        if result.multi_hand_landmarks:

            for hand_landmarks in result.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # INDEX FINGER TIP
                finger = hand_landmarks.landmark[8]

                fx = int(finger.x * w)
                fy = int(finger.y * h)

                cv2.circle(frame, (fx, fy), 12, (255,0,0), -1)

                # ---------------------------------
                # TOUCH DETECTION
                # ---------------------------------

                inside = (
                    btn_x1 < fx < btn_x2 and
                    btn_y1 < fy < btn_y2
                )

                if inside:

                    current_time = time.time()

                    # PREVENT MULTIPLE FAST TOUCHES
                    if current_time - last_touch > 1:

                        switch_on = not switch_on

                        last_touch = current_time

                        if switch_on:

                            arduino.write(b"ON\n")

                            print("✅ SYSTEM TURNED ON")

                        else:

                            arduino.write(b"OFF\n")

                            print("❌ SYSTEM TURNED OFF")

        cv2.imshow("Virtual Switch Control", frame)

        key = cv2.waitKey(1)

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------------
# MAIN
# -------------------------------
print("\n🚀 SMART MONITORING SYSTEM STARTED")

if read_sensor():

    print("\n🚨 ALERT DETECTED")

    if authenticate():

        print("\n✅ ACCESS GRANTED")

        gesture_control()

    else:

        print("\n❌ ACCESS DENIED")

else:

    print("\n✅ SYSTEM NORMAL")                                                       