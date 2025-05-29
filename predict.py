import os
import threading
import time
import cv2
import mediapipe as mp
from gtts import gTTS
import pygame
from google.cloud import vision
from ultralytics import YOLO
import tempfile

# ----- Config Google Vision API -----
KEY_PATH = r"key.json"
if not os.path.exists(KEY_PATH):
    raise FileNotFoundError(f"Google Vision API key not found at: {KEY_PATH}")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH
vision_client = vision.ImageAnnotatorClient()

# ----- Ensure screenshot folder exists -----
os.makedirs("./screenshot", exist_ok=True)

# ----- AppState Class to Replace Globals -----
class AppState:
    def __init__(self):
        self.localIP = "192.168.1.40"
        self.video = cv2.VideoCapture(f"http://{self.localIP}:81/stream")

        self.latest_frame = None  # Lưu frame mới nhất
        self.cached_data = []     # Danh sách các câu, mỗi câu là list các box-text
        self.stop_event = threading.Event()
        self.isSpeaking = False
        self.last_text = ""
        self.isReading = False
        self.isOcr = False
        self.lastTime = 0
        self.isDetected = False

# ----- TTS -----
def speak_vietnamese(text, state):
     pygame.mixer.init()
     def run_tts():
          try:
               state.isSpeaking = True
               tts = gTTS(text=text, lang='vi')
               with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    temp_filename = fp.name
                    tts.save(temp_filename)

               pygame.mixer.music.load(temp_filename)
               pygame.mixer.music.play()

               while pygame.mixer.music.get_busy():
                    time.sleep(0.1)

               pygame.mixer.quit()
               state.isSpeaking = False
          except Exception as e:
               print("TTS Error:", e)
               state.isSpeaking = False

     threading.Thread(target=run_tts, daemon=True).start()

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
            for index, lm in enumerate(landmarks.landmark):
                if index == 8:  # ngón trỏ
                    px = lm.x * width
                    py = lm.y * height
                    break
            for sentence in state.cached_data:
                # sentence: list các box-text trong 1 câu
                # check nếu trỏ nằm trong box nào của câu này
                for texts in sentence:
                    x1, y1 = texts[0][0][0], texts[0][0][1]
                    x2, y2 = texts[0][1][0], texts[0][1][1]
                    if x1 <= px <= x2 and y1 <= py <= y2:
                        if state.last_text != texts[1] and not state.isSpeaking:
                            state.last_text = texts[1]
                            speak_vietnamese(texts[1], state)
                        break


# ----- Google Vision OCR -----
def predict(state):
     def run_predict():
          state.isOcr = True
          state.isReading = True
          print("Starting Google Vision OCR")
          speak_vietnamese("Đang chạy máy đọc", state)
          time.sleep(5)

          frame = state.latest_frame
          if frame is None:
               print("Can't get latest frame")
               state.isOcr = False
               state.isReading = False
               return

          img_path = "./screenshot/cap.jpg"
          cv2.imwrite(img_path, frame)

          with open(img_path, "rb") as image_file:
               content = image_file.read()
          image = vision.Image(content=content)

          response = vision_client.text_detection(image=image)
          texts = response.text_annotations

          data = []
          if texts:
               print("Full text:", texts[0].description.strip())
               for text in texts[1:]:
                    box = [(v.x, v.y) for v in text.bounding_poly.vertices]
                    if len(box) == 4:
                         data.append([[box[0], box[2]], text.description])

          # Không gộp box, mỗi box là 1 câu riêng lẻ
          state.cached_data = [[box] for box in data]

          speak_vietnamese("Máy đọc đã chạy xong", state)

          img = cv2.imread(img_path)
          colors = [(0,255,255), (255,0,255), (255,255,0), (0,128,255), (255,128,0)]
          for idx, sentence in enumerate(state.cached_data):
               color = colors[idx % len(colors)]
               for textBox in sentence:
                    x1, y1 = textBox[0][0]
                    x2, y2 = textBox[0][1]
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
          cv2.imwrite("./screenshot/output.jpg", img)

          print("OCR cached sentences:", [[t[1] for t in sent] for sent in state.cached_data])
          state.isReading = False
          state.isOcr = False
          state.isDetected = False

     threading.Thread(target=run_predict, daemon=True).start()


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
            for box in r.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                name = class_names[cls_id]
                if name.lower() == target_object.lower() and conf >= 0.75:
                    if not state.isOcr:
                        predict(state)  # Gọi predict không chặn
                    return

# ----- Main camera & processing loop -----
def displayProcess(state):
    while state.video.isOpened() and not state.stop_event.is_set():
        ret, frame = state.video.read()
        if not ret or frame is None:
            continue
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Cập nhật frame mới nhất cho predict dùng
        state.latest_frame = frame.copy()

        edited_frame = frame.copy()
        hand_frame = frame.copy()

        if not state.isReading:
            # Vẽ vùng OCR theo từng câu với màu khác nhau
            for idx, sentence in enumerate(state.cached_data):
                color = (0,255,255)  # màu vàng
                for textBox in sentence:
                    edited_frame = cv2.rectangle(edited_frame, textBox[0][0], textBox[0][1], color, 2)

            detectFist(frame, state)
            hand_handler(hand_frame, state)

        cv2.imshow("Edited Frame", edited_frame)
        cv2.imshow("Hand Detection", hand_frame)

          key = cv2.waitKey(1)
          if key == ord('q'):
               state.stop_event.set()
               break


    state.video.release()
    cv2.destroyAllWindows()

# ----- Run app -----
if __name__ == "__main__":
    app_state = AppState()
    displayProcess(app_state)
