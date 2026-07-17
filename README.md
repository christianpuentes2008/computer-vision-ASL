# CodeVision – Real-Time ASL Fingerspelling Recognition

CodeVision is a computer vision application that recognizes **American Sign Language (ASL) fingerspelling** using a webcam. It translates hand signs into **text** and **speech** in real time, making communication easier between people who use sign language and those who do not.

---

# Features

* Real-time hand tracking using **MediaPipe**
* ASL alphabet recognition using a **Random Forest** machine learning model
* Live text captions
* Word and sentence creation
* Text-to-speech playback
* Custom training using your own hand for improved accuracy

---

# Project Files

* **`hand_tracking.py`** – Tests webcam and hand tracking by displaying the 21 hand landmarks.
* **`collect_data.py`** – Collects ASL hand-sign samples and saves them to **`asl_landmarks.csv`**.
* **`train_model.py`** – Trains the machine learning model and saves it as **`asl_model.joblib`**.
* **`live_translate.py`** – Performs live ASL recognition, builds words and sentences, and converts text into speech.

---

# Requirements

* Python 3.10–3.12
* Webcam
* Install the required libraries:

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn joblib pyttsx3
```

---

# How to Use

### **Step 1 – Test Hand Tracking**

Run:

```bash
python hand_tracking.py
```

This verifies that your webcam works and that MediaPipe can detect your hand.

---

### **Step 2 – Collect Training Data**

Run:

```bash
python collect_data.py
```

* Hold an ASL letter in front of the webcam.
* Press the matching keyboard key (A–Z).
* Repeat each letter many times from different angles and distances.
* Your data is automatically saved to **`asl_landmarks.csv`**.

---

### **Step 3 – Train the Model**

Run:

```bash
python train_model.py
```

The program trains the machine learning model, displays its accuracy, and saves the trained model as **`asl_model.joblib`**.

---

### **Step 4 – Start Live Translation**

Run:

```bash
python live_translate.py
```

The application recognizes ASL letters in real time, builds words and sentences, and reads completed sentences aloud.

---

# Keyboard Controls

 Key            Function                     
 -------------  ---------------------------- 
 **Space**      Finish the current word      
 **Backspace**  Delete the previous letter   
 **Enter**      Speak the completed sentence 
 **C**          Clear all text               
 **ESC**        Exit the application         

---

# things I did to better my results

* Use bright lighting.
* Keep your entire hand inside the camera frame.
* Use a plain background.
* Hold each sign steady for about one second.
* Collect many training samples for every letter.
* Retrain the model after adding new data for better accuracy.

---

# Future Improvements

* Add support for motion-based letters (**J** and **Z**).
* Recognize complete ASL words and phrases.
* Support two-hand gestures.
* Integrate live captions into video conferencing applications such as Zoom or Microsoft Teams.

---

# License

Choose a license for your project (such as the MIT License) before sharing or publishing CodeVision.
