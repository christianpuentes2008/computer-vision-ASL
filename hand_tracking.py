import cv2
import mediapipe as mp
import csv
import os

# --- Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,          # one hand at a time for cleaner training data
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

DATA_FILE = "asl_landmarks.csv"

# Letters we're collecting (J and Z need motion, so skip for now)
LETTERS = [c for c in "ABCDEFGHIKLMNOPQRSTUVWXY"]

# Create CSV with header if it doesn't exist yet
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"]
        for i in range(21):  # 21 hand landmarks
            header += [f"x{i}", f"y{i}", f"z{i}"]
        writer.writerow(header)

camera = cv2.VideoCapture(0)

print("Instructions:")
print("- Hold up the hand shape for a letter")
print("- Press the letter key (A-Y, no J/Z) to save that frame as a sample")
print("- Press ESC to quit")
print("- Aim for at least 30-50 samples per letter, varying angle/distance slightly")

while True:
    success, frame = camera.read()
    if not success:
        print("Failed to access webcam.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    landmarks_flat = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Flatten landmarks into a single list [x0,y0,z0, x1,y1,z1, ...]
            landmarks_flat = []
            for lm in hand_landmarks.landmark:
                landmarks_flat += [lm.x, lm.y, lm.z]

    # Show instructions on screen
    cv2.putText(frame, "Press A-Y to label, ESC to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("CodeVision - Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    if landmarks_flat is not None:
        pressed_letter = chr(key).upper() if 65 <= key <= 122 else None
        if pressed_letter in LETTERS:
            with open(DATA_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([pressed_letter] + landmarks_flat)
            print(f"Saved sample for letter: {pressed_letter}")

camera.release()
cv2.destroyAllWindows()
print(f"Data saved to {DATA_FILE}")