import cv2
import mediapipe as mp
import joblib
import pyttsx3
from collections import deque, Counter

MODEL_FILE = "asl_model.joblib"

# --- Load trained model ---
model = joblib.load(MODEL_FILE)

# --- MediaPipe setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# --- Text-to-speech setup ---
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 160)

def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# --- Prediction stabilization ---
PREDICTION_BUFFER_SIZE = 15
CONFIDENCE_THRESHOLD = 0.6
recent_predictions = deque(maxlen=PREDICTION_BUFFER_SIZE)

last_accepted_letter = None
letter_hold_frames = 0
HOLD_FRAMES_REQUIRED = 20

sentence = ""
current_word = ""

camera = cv2.VideoCapture(0)

print("Controls:")
print("- Hold a letter shape steady to type it")
print("- SPACE = add space between words")
print("- BACKSPACE = delete last character")
print("- ENTER = speak the current sentence")
print("- C = clear sentence")
print("- ESC = quit")

while True:
    success, frame = camera.read()
    if not success:
        print("Failed to access webcam.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    predicted_letter = None
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks_flat = []
            for lm in hand_landmarks.landmark:
                landmarks_flat += [lm.x, lm.y, lm.z]

            probs = model.predict_proba([landmarks_flat])[0]
            best_idx = probs.argmax()
            predicted_letter = model.classes_[best_idx]
            confidence = probs[best_idx]

    if predicted_letter is not None and confidence >= CONFIDENCE_THRESHOLD:
        recent_predictions.append(predicted_letter)
    else:
        recent_predictions.append(None)

    if len(recent_predictions) == PREDICTION_BUFFER_SIZE:
        most_common, count = Counter(recent_predictions).most_common(1)[0]
        stable_ratio = count / PREDICTION_BUFFER_SIZE

        if most_common is not None and stable_ratio >= 0.7:
            if most_common == last_accepted_letter:
                letter_hold_frames += 1
            else:
                letter_hold_frames = 1
                last_accepted_letter = most_common

            if letter_hold_frames == HOLD_FRAMES_REQUIRED:
                current_word += most_common
                recent_predictions.clear()
                letter_hold_frames = 0
        else:
            last_accepted_letter = None
            letter_hold_frames = 0

    display_text = f"Live: {predicted_letter or '-'} ({confidence:.0%})"
    cv2.putText(frame, display_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"Word: {current_word}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.putText(frame, f"Sentence: {sentence}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, "SPACE=space  BKSP=delete  ENTER=speak  C=clear  ESC=quit",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("CodeVision - ASL Translation", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    elif key == 32:
        if current_word:
            sentence += current_word + " "
            current_word = ""
    elif key == 8:
        if current_word:
            current_word = current_word[:-1]
        elif sentence:
            sentence = sentence.rstrip()[:-1]
    elif key == 13:
        full_text = (sentence + current_word).strip()
        if full_text:
            print(f"Speaking: {full_text}")
            speak(full_text)
    elif key in (ord('c'), ord('C')):
        sentence = ""
        current_word = ""

camera.release()
cv2.destroyAllWindows()