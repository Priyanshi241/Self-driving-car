import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import math
import argparse
from google.colab.patches import cv2_imshow
#import RPi.GPIO as GPIO

# Define GPIO pin numbers
# TRIG = 17
# ECHO = 27
# led = 22
# m11 = 16
# m12 = 12
# m21 = 21
# m22 = 20

# GPIO setup
# GPIO.setwarnings(False)
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(TRIG, GPIO.OUT)
# GPIO.setup(ECHO, GPIO.IN)
# GPIO.setup(led, GPIO.OUT)
# GPIO.setup(m11, GPIO.OUT)
# GPIO.setup(m12, GPIO.OUT)
# GPIO.setup(m21, GPIO.OUT)
# GPIO.setup(m22, GPIO.OUT)

model = load_model('/content/full_CNN_model.h5')
cascadePath = r'/content/trafficLightCascade.xml'
cascade = cv2.CascadeClassifier(cascadePath)

class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

def slope(vx1, vx2, vy1, vy2):
    m = float(vy2 - vy1) / float(vx2 - vx1)
    theta1 = math.atan(m)
    return theta1 * (180 / np.pi)

def road_lines(image, lanes):
    small_img = cv2.resize(image, (160, 80))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    prediction = model.predict(small_img)[0] * 255
    lanes.recent_fit.append(prediction)
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis=0)

    lane_center = np.argmax(lanes.avg_fit, axis=1)
    center_position = len(lane_center) // 2
    lane_position = np.argmax(lane_center)

    distance_from_center = lane_position - center_position

    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))
    lane_image = cv2.resize(lane_drawn, (image.shape[1], image.shape[0]))
    lane_image = lane_image.astype(image.dtype)

    result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    if distance_from_center > 10:
        print("Turn right")
    elif distance_from_center < -10:
        print("Turn left")

    return result


def identifyDominantColor(roi):
    pixels = roi.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
    _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    dominant_color = centers[np.argmax(np.unique(labels, return_counts=True)[1])]
    if dominant_color[0] > 150 and dominant_color[1] < 100 and dominant_color[2] < 100:
        return 'red'
    elif dominant_color[0] < 100 and dominant_color[1] > 150 and dominant_color[2] < 100:
        return 'green'
    elif dominant_color[0] < 100 and dominant_color[1] < 100 and dominant_color[2] > 150:
        return 'blue'
    else:
        return 'unknown'


def detect_traffic_light(frame):
    red_light_detected = False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    traffic_lights = cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in traffic_lights:
        roi = frame[y:y+h, x:x+w]
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        color = identifyDominantColor(rgb_roi)
        if color == 'red':
            return True
    return False

# def stop():
#   print('stop')
#   GPIO.output(m11, 0)
#   GPIO.output(m12, 0)
#   GPIO.output(m21, 0)
#   GPIO.output(m22, 0)

# def forward():
#   GPIO.output(m11, 0)
#   GPIO.output(m12, 1)
#   GPIO.output(m21, 1)
#   GPIO.output(m22, 0)
#   print('Forward')


def process_video_with_object_detection(frame):
    CLASSES = ["boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor", "box", "keys", "bottle"]

    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe('/content/MobileNetSSD_deploy.prototxt.txt', '/content/MobileNetSSD_deploy (2).caffemodel')

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def process_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        lanes = Lanes()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            red_light_detected = detect_traffic_light(frame)
            object_detected = process_video_with_object_detection(frame)
            if  object_detected and red_light_detected:
                print("Stop the car: Red traffic light detected and object detected in the path of the car.")
                # stop()
            elif red_light_detected:
                print("Stop the car: Red traffic light detected.")
                # stop()
            elif object_detected:
                print("Object detected in the path of the car. Taking necessary action...")
                # stop()
            else:
                print("No red traffic light detected and no object in the path of the car. Checking for lane detection... ")
                #print("If lane detection is not working than uncomment forward and comment everything else  ")
                # forward
                lane_detected_frame = road_lines(frame, lanes)
                cv2_imshow(lane_detected_frame)

            cv2_imshow(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("An error occurred:", e)


if __name__ == '__main__':
    video_path = input("Enter the path of the video: ")
    process_video(video_path)
