import cv2
import numpy as np
import argparse
from pygame import mixer
#
# parser = argparse.ArgumentParser()
# args, unknown = parser.parse_known_args()


# Load yolo
def load_yolo():
    net = cv2.dnn.readNet("Files/yolov3.weights", "Files/yolov3.cfg")
    classes = []
    with open("Files/obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(10, 3))
    return net, classes, colors, output_layers


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_DUPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
            if label == 'Fire':
                mixer.init()
                mixer.music.load("Alarms/Fire.mpeg")
                mixer.music.play()

            elif label == 'Rifle':
                mixer.init()
                mixer.music.load("Alarms/Rifle.mpeg")
                mixer.music.play()

            elif label == 'Gun':
                mixer.init()
                mixer.music.load("Alarms/Gun.mpeg")
                mixer.music.play()

    img = cv2.resize(img, (800, 600))

    cv2.imshow("Image", img)


def webcam_detect():
    model, classes, colors, output_layers = load_yolo()
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cap.release()


print('---- Starting Web Cam object detection ----')
webcam_detect()
cv2.destroyAllWindows()
