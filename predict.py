import os
import threading
import time
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vietocr.vietocr.tool.predictor import Predictor
from vietocr.vietocr.tool.config import Cfg

import easyocr
FONT = './PaddleOCR/doc/fonts/latin.ttf'

ocrCall = False
cached_data = []
stop_event = threading.Event()

def putVietnameseText(frame, text, position):
    font_path = "arial.ttf"
    font_size = 15
    color = (255, 0, 0)

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype(font_path, font_size)

    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)

    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return frame

mpHands = mp.solutions.hands 
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

def hand_handler(frame):
     global cached_data
     height, width = frame.shape[:2]
     results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
     last_text = ""
     if results.multi_hand_landmarks:
          for landmarks in results.multi_hand_landmarks:
               mpDraw.draw_landmarks(frame, landmarks, mpHands.HAND_CONNECTIONS)
               px= 0
               py= 0
               for index, lm in enumerate(landmarks.landmark):
                    if index == 8:
                         px = lm.x * width
                         py = lm.y * height
                         break
               for texts in cached_data:
                    x1, y1 = texts[0][0][0], texts[0][0][1]
                    x2, y2 = texts[0][1][0], texts[0][1][1]
                    if (x1 <= px <= x2 and y1 <= py <= y2):
                         print(f"Text in finger: {texts[1]}")


def predict():
     dpi = 100
     padding = 0

     global cached_data, ocrCall
     # Configure of VietOCR
     # Default weight
     config = Cfg.load_config_from_name('vgg_transformer')
     # Custom weight
     # config = Cfg.load_config_from_file('vi00_vi01_transformer.yml')
     # config['weights'] = './pretrain_ocr/vi00_vi01_transformer.pth'

     config['cnn']['pretrained'] = True
     config['predictor']['beamsearch'] = True
     config['device'] = 'cpu'

     recognitor = Predictor(config)
     reader = easyocr.Reader(['vi'], gpu=True)

     while not stop_event.is_set():
          if not ocrCall:
               time.sleep(1)
               continue
          # Text detection
          print("Starting AI...")
          img_path = "./screenshot/cap.jpg"
          img = cv2.imread(img_path)
   
          # result = detector.ocr(img_path, cls=False, det=True, rec=False)
          # result = result[:][:][0]
          result = reader.readtext(img)

          # Filter Boxes
          boxes = []
          for (box, text, confidence) in result:
               boxes.append([[int(box[0][0]), int(box[0][1])], [int(box[2][0]), int(box[2][1])]])

          # Add padding to boxes
          for box in boxes:
               box[0][0] = box[0][0] - padding
               box[0][1] = box[0][1] - padding
               box[1][0] = box[1][0] + padding
               box[1][1] = box[1][1] + padding

          # Text recognizion
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
                    print("Error")
                    break
               rec_result = recognitor.predict(cropped_image)

               text = rec_result#[0]
               data.append([[(box[0][0], box[0][1]), (box[1][0], box[1][1])], text])
          
          cached_data = data
          print(cached_data)

          #Debugging image
          color = (0, 255, 255)
          for textBox in cached_data:
               edited_frame = cv2.rectangle(img, textBox[0][0], textBox[0][1], color, 2)
               edited_frame = putVietnameseText(img, textBox[1], textBox[0][0])
          cv2.imwrite("./screenshot/output.jpg", edited_frame)
          #-------------------------------------------

          ocrCall = False

def displayProcess():
     global ocrCall
     #Config of camera
     data = np.load('calibration_data.npz')
     mtx = data['camera_matrix']
     dist = data['dist_coeffs']
     rvecs = data['rvecs']
     tvecs = data['tvecs']

     localIP = "192.168.1.37"
     video = cv2.VideoCapture(f"http://{localIP}:81/stream")
     width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
     height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

     while video.isOpened() and not stop_event.is_set():
          ret, frame = video.read()
          if not ret or frame is None: continue
          frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
          edited_frame = frame.copy()
          hand_frame = frame.copy()
          
          color = (0, 255, 255)
          for textBox in cached_data:
               edited_frame = cv2.rectangle(edited_frame, textBox[0][0], textBox[0][1], color, 2)
               #edited_frame = putVietnameseText(edited_frame, textBox[1], textBox[0][0])
          
          hand_handler(hand_frame)
          cv2.imshow("ESP32 Camera OCR", edited_frame)
          cv2.imshow("Hand", hand_frame)

          key = cv2.waitKey(1)
          if key == ord('q'):
               stop_event.set()
               break
          elif key == ord('c'):
               cv2.imwrite("./screenshot/cap.jpg", frame)
               ocrCall = True


display_thread = threading.Thread(target=displayProcess)
ocr_thread = threading.Thread(target=predict)

display_thread.start()
ocr_thread.start()

display_thread.join()
ocr_thread.join()