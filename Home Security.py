import cv2
import numpy as np
from PIL import Image
import os
import face_recognition
import argparse
from pygame import mixer

face_detector = cv2.CascadeClassifier('Files/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


def Dataset():
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    # For each person, enter one numeric face id
    face_id = input('\n enter new name end press enter ')

    print("\nInitializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    count = 0

    while True:

        ret, img = cam.read()
        img = cv2.flip(img, 1)  # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1

            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/" + str(face_id) + ".jpg", gray[y:y + h, x:x + w])

            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 1:  # Take 1 face sample and stop video
            break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()


def getImagesAndLabels(path):
    path = 'dataset/'
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
        img_numpy = np.array(PIL_img, 'uint8')

        id = os.path.split(imagePath)[-1].split(".")[0]
        faces = face_detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids


def Face_Recognition():
    cap = cv2.VideoCapture(0)

    # Load a sample picture and learn how to recognize it.
    # riya_image = face_recognition.load_image_file("dataset/Riya.jpg")
    # riya_face_encoding = face_recognition.face_encodings(riya_image)[0]
    # #
    # # # Load a second sample picture and learn how to recognize it.
    # sam_image = face_recognition.load_image_file("dataset/Trump.jpg")
    # sam_face_encoding = face_recognition.face_encodings(sam_image)[0]
    # temp_image = face_recognition.load_image_file(getImagesAndLabels('dataset'))
    # temp_face_encoding = face_recognition.face_encodings(temp_image)[0]
    # frame, id = getImagesAndLabels('dataset')
    # known_face_encodings = []
    # known_face_names = []
    # for i in id:
    #     temp_img = face_recognition.load_image_file(i)
    #     temp_face_encoding = face_recognition.face_encodings(temp_img)[0]
    #     known_face_encodings.append(temp_face_encoding)
    #     known_face_names.append(i.capitalize())
    model, classes, colors, output_layers = load_yolo()
    known_face_encodings = []
    known_face_names = []

    path = 'dataset/'
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        # PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
        # img_numpy = np.array(PIL_img, 'uint8')
        # print(imagePath)
        temp_img = face_recognition.load_image_file(imagePath)
        print(face_recognition.face_encodings(temp_img))
        temp_face_encoding = face_recognition.face_encodings(temp_img)[0]
        known_face_encodings.append(temp_face_encoding)
        id1 = os.path.split(imagePath)[-1].split(".")[0]
        known_face_names.append(id1.capitalize())

    # Create arrays of known face encodings and their names

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, frame)
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        #
        # # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        # rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    if name == "Unknown":
                        mixer.init()
                        mixer.music.load("Alarms/taunt.wav")
                        mixer.music.play()
                    if name != "Unknown":
                        mixer.init()
                        mixer.music.load("Alarms/alarm.wav")
                        mixer.music.play()

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    cap.release()
    cv2.destroyAllWindows()


parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()


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
                mixer.music.load("Alarms/alarm.wav")
                mixer.music.play()

            elif label == 'Rifle':
                mixer.init()
                mixer.music.load("Alarms/taunt.wav")
                mixer.music.play()

            elif label == 'Gun':
                mixer.init()
                mixer.music.load("Alarms/CantinaBand60.wav")
                mixer.music.play()

    img = cv2.resize(img, (800, 600))


print('---- Starting Web Cam Home Security ----')
ans = input("Do you want to add new face [y/n]: \n")
if ans.lower() == 'y':
    Dataset()
    faces, ids = getImagesAndLabels('dataset/')
    ans1 = input("Do you want to check if your face is saved or not? [y/n]: \n")
    if ans1.lower() == 'y':
        Face_Recognition()
else:
    Face_Recognition()
