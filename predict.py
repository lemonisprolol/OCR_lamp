import os
import threading
import time
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

from vietocr.vietocr.tool.predictor import Predictor
from vietocr.vietocr.tool.config import Cfg
import tempfile
from gtts import gTTS
import pygame

import easyocr

from ultralytics import YOLO

# ----- AppState Class to Replace Globals -----
class AppState:
     def __init__(self):
          self.localIP = "192.168.94.176"
          self.video = cv2.VideoCapture(f"http://{self.localIP}:81/stream")

          self.cached_data = []
          self.stop_event = threading.Event()
          self.isSpeaking = False
          self.last_text = ""
          self.isReading = False
          self.isOcr = False
          self.lastTime = 0
          self.isDetected = False

# ----- TTS -----
def speak_vietnamese(text, state):
     try:
          state.isSpeaking = True
          tts = gTTS(text=text, lang='vi')
          with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
               temp_filename = fp.name
               tts.save(temp_filename)

          pygame.mixer.init()
          pygame.mixer.music.load(temp_filename)
          pygame.mixer.music.play()

          while pygame.mixer.music.get_busy():
               time.sleep(0.1)

          state.isSpeaking = False
          pygame.mixer.quit()
     except Exception as e:
          print("TTS Error:", e)

# ----- Hand Detection (MediaPipe) -----
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

def hand_handler(frame, state):
     height, width = frame.shape[:2]
     results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
     if results.multi_hand_landmarks:
          for landmarks in results.multi_hand_landmarks:
               mpDraw.draw_landmarks(frame, landmarks, mpHands.HAND_CONNECTIONS)
               px = 0
               py = 0
               for index, lm in enumerate(landmarks.landmark):
                    if index == 8:
                         px = lm.x * width
                         py = lm.y * height
                         break
               for texts in state.cached_data:
                    x1, y1 = texts[0][0][0], texts[0][0][1]
                    x2, y2 = texts[0][1][0], texts[0][1][1]
                    if (x1 <= px <= x2 and y1 <= py <= y2):
                         print(f"Text in finger: {texts[1]}")
                         if state.last_text != texts[1] and not state.isSpeaking:
                              state.last_text = texts[1]
                              threading.Thread(target=speak_vietnamese, args=(texts[1], state)).start()


# ----- OCR Thread -----
config = Cfg.load_config_from_name('vgg_transformer')
config['cnn']['pretrained'] = True
config['predictor']['beamsearch'] = True
config['device'] = 'cpu'

recognitor = Predictor(config)
reader = easyocr.Reader(['vi'], gpu=True)
def predict(state):
     state.isOcr = True
     print("Starting AI OCR")
     speak_vietnamese("Đang chạy máy đọc", state)
     ret, frame = state.video.read()
     if not ret:
          print("❌ Failed to grab frame after delay")
          state.isOcr = False
          return
     frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
     cv2.imwrite("./screenshot/cap.jpg", frame)

     state.isReading = True
     img_path = "./screenshot/cap.jpg"
     img = cv2.imread(img_path)

     result = reader.readtext(img, width_ths=1)

     boxes = []
     for (box, text, confidence) in result:
          boxes.append([[int(box[0][0]), int(box[0][1])], [int(box[2][0]), int(box[2][1])]])

     data = []
     h, w = img.shape[:2]
     for box in boxes:
          x1, y1 = max(1, box[0][0]), max(1, box[0][1])
          x2, y2 = min(w, box[1][0]), min(h, box[1][1])
          cropped_image = img[y1:y2, x1:x2]
          if cropped_image.size == 0 or cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
               continue

          try:
               cropped_image = Image.fromarray(cropped_image)
          except:
               print("Error converting crop to PIL")
               break

          rec_result = recognitor.predict(cropped_image)
          data.append([[(box[0][0], box[0][1]), (box[1][0], box[1][1])], rec_result])

     state.cached_data = data
     speak_vietnamese("Máy đọc đã chạy xong", state)
     state.isReading = False
     print("OCR Cached data:", state.cached_data)

     color = (0, 255, 255)
     for textBox in state.cached_data:
          img = cv2.rectangle(img, textBox[0][0], textBox[0][1], color, 2)
     cv2.imwrite("./screenshot/output.jpg", img)
     state.isOcr = False
     state.isDetected = False


# ----- YOLO fist detection setup -----
yolo_model = YOLO("fist.pt")
class_names = yolo_model.names
target_object = "fist"

def detectFist(frame, state):
     yolo_results = yolo_model.predict(source=frame, conf=0.5, verbose=False)

     if yolo_results:
          for r in yolo_results:
               boxes = r.boxes
               if boxes is None:
                    continue
               for box in boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    name = class_names[cls_id]

                    if name.lower() != target_object.lower() or conf < 0.7:
                         continue
                    if not state.isOcr:
                         ocr_thread = threading.Thread(target=predict, args=(state,))
                         ocr_thread.start()
                    return


# ----- Main camera & processing loop -----
def displayProcess(state):


     while state.video.isOpened() and not state.stop_event.is_set():
          ret, frame = state.video.read()
          if not ret or frame is None:
               continue
          frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
          edited_frame = frame.copy()
          hand_frame = frame.copy()

          if not state.isReading:
               color = (0, 255, 255)
               for textBox in state.cached_data:
                    edited_frame = cv2.rectangle(edited_frame, textBox[0][0], textBox[0][1], color, 2)

               detectFist(frame, state)
               hand_handler(hand_frame, state)

          cv2.imshow("ESP32 Camera OCR + Fist Detection", edited_frame)
          cv2.imshow("Hand", hand_frame)

          key = cv2.waitKey(1)
          if key == ord('q'):
               state.stop_event.set()
               break


# ----- Start Threads -----
if __name__ == "__main__":
     state = AppState()
     display_thread = threading.Thread(target=displayProcess, args=(state,))
     display_thread.start()
     display_thread.join()

