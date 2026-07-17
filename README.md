# CodeVision

CodeVision is a computer vision project that recognizes American Sign Language (ASL) fingerspelling using a webcam. It translates hand signs into text and speech in real time, making communication easier between people who use sign language and those who do not.

## Features

* Real-time hand tracking using MediaPipe
* ASL alphabet recognition using a Random Forest machine learning model
* Live text captions
* Word and sentence creation
* Text-to-speech playback
* Custom training using your own hand for improved accuracy

## Project Files

hand_tracking.py

Tests the webcam and hand tracking by displaying the 21 hand landmarks.

collect_data.py

Collects ASL hand-sign samples and saves them to asl_landmarks.csv.

train_model.py

Trains the machine learning model and saves it as asl_model.joblib.

live_translate.py

Recognizes ASL letters in real time, builds words and sentences, and converts text into speech.

## Requirements

* Python 3.10 to 3.12
* A webcam
* The required Python libraries are installed

Install the required libraries by running:

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn joblib pyttsx3
```

## How to Use

Step 1: Test Hand Tracking

Run:

```bash
python hand_tracking.py
```

This program opens your webcam and checks that MediaPipe can detect your hand correctly by displaying the 21 hand landmarks.

Step 2: Collect Training Data

Run:

```bash
python collect_data.py
```

Hold an ASL letter in front of the webcam and press the matching key on your keyboard. The program saves the letter and its hand landmark data into asl_landmarks.csv. Repeat this process for every letter several times while changing the angle and distance of your hand to improve the model's accuracy.

Step 3: Train the Model

Run:

```bash
python train_model.py
```

The program loads the collected data, trains a Random Forest classifier, displays the model's accuracy, and saves the trained model as asl_model.joblib.

Step 4: Start Live Translation

Run:

```bash
python live_translate.py
```

The application opens your webcam and begins recognizing ASL letters in real time. As letters are recognized, they are combined into words and sentences that can be spoken aloud using text-to-speech.

## Keyboard Controls

* Space: Finish the current word.
* Backspace: Delete the previous letter.
* Enter: Speak the completed sentence.
* C: Clear all text.
* Esc: Close the application.

## Tips

* Use good lighting.
* Keep your entire hand inside the camera frame.
* Use a plain background whenever possible.
* Hold each sign steady for about one second.
* Collect many examples for each letter before training.
* Retrain the model whenever you add more training data.

## Future Improvements

* Add support for the motion-based letters J and Z.
* Recognize complete ASL words and phrases instead of only fingerspelling.
* Support two-hand gestures.
* Integrate live captions into video conferencing applications such as Zoom or Microsoft Teams.

