from flask import Flask, request, render_template, send_file, Response
from werkzeug.utils import secure_filename
import io
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import os

from datetime import datetime
#from database import insert_violation_log
"""
from playsound import playsound
from email_alert import send_email_alert
from line_notify import send_line_notify
"""
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'


class Detection:
    def __init__(self):
        #download weights from here:https://github.com/ultralytics/ultralytics and change the path
        self.model = YOLO(r"mocyclehelmet.pt")

    def predict(self, img, classes=[], conf=0.5):
        if classes:
            results = self.model.predict(img, classes=classes, conf=conf)
        else:
            results = self.model.predict(img, conf=conf)

        return results

    def predict_and_detect(self, img, classes=[], conf=0.6, rectangle_thickness=1, text_thickness=1):
        results = self.predict(img, classes, conf=conf)
        count_human = 0
        count_motorcycle = 0
        count_no_helmet = 0
        count_helmet = 0
        for result in results:
            for box in result.boxes:
                if box.cls==0:
                    label = "HUMAN"
                    count_human += 1
                    strCon = '{:.2f}'.format(float(box.conf))
                    cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (96, 255, 51), rectangle_thickness)
                    cv2.putText(img, label+ ' ' + strCon, (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 1), cv2.FONT_HERSHEY_PLAIN, 1, (96, 255, 51), text_thickness, cv2.LINE_AA)

                elif box.cls==1:
                    label = "ON-MOTORCYCLE"
                    strCon = '{:.2f}'.format(float(box.conf))
                    cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 218, 51), rectangle_thickness)
                    cv2.putText(img, label+ ' ' + strCon, (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 9), cv2.FONT_HERSHEY_PLAIN, 1, (255, 218, 51), text_thickness, cv2.LINE_AA)

                elif box.cls==2:
                    label = "MOTORCYCLE"
                    count_motorcycle += 1
                    strCon = '{:.2f}'.format(float(box.conf))
                    cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (2, 51, 250), rectangle_thickness)
                    cv2.putText(img, label+ ' ' + strCon, (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 1), cv2.FONT_HERSHEY_PLAIN, 1, (2, 51, 250), text_thickness, cv2.LINE_AA)

                elif box.cls==3:
                    label = "NO-HELMET"
                    count_no_helmet += 1
                    strCon = '{:.2f}'.format(float(box.conf))
                    cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (166, 51, 255), rectangle_thickness)
                    cv2.putText(img, label+ ' ' + strCon, (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 9), cv2.FONT_HERSHEY_PLAIN, 1, (166, 51, 255), text_thickness, cv2.LINE_AA)
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    img_path = f'snapshots/no_helmet_{timestamp}.jpg'
                    cv2.imwrite(img_path, img)

                    # insert_violation_log(timestamp, img_path)
                    # playsound('static/alert.wav', block=False)
                    # send_email_alert(img_path, timestamp)
                    # send_line_notify(img_path, timestamp)

                elif box.cls==4:
                    label = "HELMET"
                    count_helmet += 1
                    strCon = '{:.2f}'.format(float(box.conf))
                    cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (51, 239, 255), rectangle_thickness)
                    cv2.putText(img, label+ ' ' + strCon, (int(box.xyxy[0][0]), int(box.xyxy[0][1]) + 13), cv2.FONT_HERSHEY_PLAIN, 1, (51, 239, 255), text_thickness, cv2.LINE_AA)

                else:
                    label = "NOT-CLASS"
                    cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
                    cv2.putText(img, label+ ' ' + strCon, (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 9), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
                
                #label = f"{classes[i]}: {class_names[classes[i]]} ({con[i]:.2f})"
                #cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
                #cv2.putText(img, f"{result.names[int(box.cls[0])]}"+ ' ' + str(box.conf.round(decimals= 2)), (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 9), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), text_thickness)
                #cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
      
        blue = (183, 183, 183)
        cv2.rectangle(img, (0,0), (150, 92), blue, -1)
        cv2.putText(img, f"MotorCycle: {count_motorcycle}", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (2, 51, 250), 1)
        cv2.putText(img, f"Human: {count_human}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (96, 255, 51), 1)
        cv2.putText(img, f"No Helmet: {count_no_helmet}", (10, 60), cv2.FONT_HERSHEY_PLAIN, 1, (166, 51, 255), 1)
        cv2.putText(img, f"Use Helmet: {count_helmet}", (10, 80), cv2.FONT_HERSHEY_PLAIN, 1, (51, 239, 255), 1)

        return img, results

    def detect_from_image(self, image):
        result_img, _ = self.predict_and_detect(image, classes=[], conf=0.6)
        
        return result_img

detection = Detection()


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/object_detect1/', methods=['POST'])
def apply_detection():
    if 'image' not in request.files:
        return 'No file part'

    file = request.files['image']
    if file.filename == '':
        return 'No selected file'

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img = Image.open(file_path).convert("RGB")
        img = np.array(img)
        img = cv2.resize(img, (712, 512))
        img = detection.detect_from_image(img)
        output = Image.fromarray(img)

        buf = io.BytesIO()
        output.save(buf, format="PNG")
        buf.seek(0)

        os.remove(file_path)
        return send_file(buf, mimetype='image/png')


@app.route('/video')
def index_video():
    return render_template('video.html')


def gen_frames():
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('rtsp://username:password@192.168.10.42/1')
    #cap = cv2.VideoCapture('rtsp://admin:Tip12345@192.168.10.42/H264?ch=1&subtype=0')
    #rtsp://admin:pas12345@192.168.11.110:554/Streaming/Channels/202/
    #cap = cv2.VideoCapture()
    #cap.open("rtsp://admin:Tip12345@192.168.10.42:554/Streaming/Channels/2/")
    
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (612, 512))
        if frame is None:
            break
        frame = detection.detect_from_image(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
    #app.run(debug=True)
    #http://localhost:8000/video for video source
    #http://localhost:8000 for image sourcep