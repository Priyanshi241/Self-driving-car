import numpy as np
import cv2
import picamera
import io
import time
import RPi.GPIO as GPIO

# Define GPIO pin numbers
ENA=17
IN1=27
IN2=22

# GPIO setup
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)


# Load traffic light cascade classifier
cascadePath = r'/content/trafficLightCascade.xml'
cascade = cv2.CascadeClassifier(cascadePath)


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
    # Function to detect traffic light and return signal
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

def process_video_with_object_detection(frame):
    # Initialize the list of class labels MobileNet SSD was trained to detect
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor", "box", "keys", "bottle","black box"]

    # Load MobileNet SSD model
    net = cv2.dnn.readNetFromCaffe('/content/MobileNetSSD_deploy.prototxt.txt', '/content/MobileNetSSD_deploy.caffemodel')

    # Grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.2:
            # Extract the index of the class label from the detections
            idx = int(detections[0, 0, i, 1])

            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box around the detected object
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Draw the class label and confidence
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # If object detected, return True, else return False
    return True if detections.shape[2] > 0 else False


def stop():
  GPIO.output(m11, 0)
  GPIO.output(m12, 0)
  GPIO.output(m21, 0)
  GPIO.output(m22, 0)

def forward():
  GPIO.output(ENA, GPIO.HIGH)
  GPIO.output(IN1, GPIO.HIGH)
  GPIO.output(IN2, GPIO.LOW)

def process_video(video_data):
    try:
        stream = io.BytesIO(video_data)
        cap = cv2.VideoCapture(stream)
        lanes = Lanes()

        while True:
            # Capture the first 10 seconds of video
            start_time = time.time()
            while time.time() - start_time < 10:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect traffic light
                red_light_detected = detect_traffic_light(frame)

                # Object detection
                object_detected = process_video_with_object_detection(frame)

                # Lane detection
                if  object_detected and red_light_detected:
                  stop()
                elif red_light_detected:
                  stop()
                elif object_detected:
                  stop()
                else:
                  forward()

                # Detect Final light
                green_light_detected = detect_traffic_light(frame)
                if green_light_detected:
                    stop()

                # Display the original frame
                cv2.imshow(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("An error occurred:", e)

if __name__ == '__main__':
    # Capture video from Raspberry Pi camera
    with picamera.PiCamera() as camera:
        camera.resolution = (1296,972)
        while True:
            camera.start_recording('video.h264')
            camera.wait_recording(10)  # Record for 10 seconds (you can adjust this duration)
            camera.stop_recording()

            time.sleep(2)
