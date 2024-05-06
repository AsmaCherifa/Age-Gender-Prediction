import cv2
import numpy as np
import math
import argparse
from flask import Flask, render_template, Response, request
from PIL import Image
import io
from tensorflow.keras.models import load_model

UPLOAD_FOLDER = './UPLOAD_FOLDER'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    # Grab the frame dimensions and convert it to a blob.
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [
                                 104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):    # Looping over the detections.
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:   # Compare it to the confidence threshold.
            x1 = int(detections[0, 0, i, 3]*frameWidth)
            y1 = int(detections[0, 0, i, 4]*frameHeight)
            x2 = int(detections[0, 0, i, 5]*frameWidth)
            y2 = int(detections[0, 0, i, 6]*frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes

parser = argparse.ArgumentParser()
parser.add_argument('--image')

args = parser.parse_args()


age_model = load_model('age_model.keras')

# this function is able to generate a frame on the photo
def gen_frames():
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"

    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    faceNet = cv2.dnn.readNet(faceModel, faceProto)

    video = cv2.VideoCapture(0)
    padding = 20
    while cv2.waitKey(1) < 0:
        hasFrame, frame = video.read()
        if not hasFrame:
            cv2.waitKey()
            break

        resultImg, faceBoxes = highlightFace(faceNet, frame)
        if not faceBoxes:
            print("No face detected")
            continue

        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1]-padding):min(faceBox[3]+padding, frame.shape[0]-1),
                         max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

            # Convert face to grayscale
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            # Resize face to (128, 128)
            face_resized = cv2.resize(face_gray, (128, 128))
            # Expand dimensions to match model input shape
            face_input = np.expand_dims(face_resized, axis=-1)
            face_input = np.expand_dims(face_input, axis=0)

            # Predict gender and age
            gender_prob, age_pred = age_model.predict(face_input)

            # Convert predicted indices to integers
            predicted_gender_index = np.argmax(gender_prob)
            predicted_age_index = np.argmax(age_pred)

            # Retrieve gender and age labels
            predicted_gender = genderList[predicted_gender_index]
            predicted_age = ageList[predicted_age_index]

            print("Gender:", predicted_gender)
            print("Age:", predicted_age)

            cv2.putText(resultImg, f'{predicted_gender}, {predicted_age}', (faceBox[0], faceBox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        ret, encodedImg = cv2.imencode('.jpg', resultImg)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')

# this function is able to generate a frame on the vd
def gen_frames_photo(img_file, age_model):
    # Load the face detection model
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    faceNet = cv2.dnn.readNet(faceModel, faceProto)

    # Convert the input image to RGB
    frame = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
    padding = 20

    # Detect faces in the input image
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:   # If no faces are detected
        print("No face detected")   # Then it will print this message
        return None

    for faceBox in faceBoxes:
        # Extract face region from the image
        face = frame[max(0, faceBox[1]-padding):min(faceBox[3]+padding, frame.shape[0]-1), 
                     max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

        # Resize and convert to grayscale
        face_resized = cv2.resize(face, (128, 128))
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
        face_gray = np.expand_dims(face_gray, axis=-1)  # Add single channel dimension
        face_gray = np.expand_dims(face_gray, axis=0)   # Add batch dimension

        # Predict gender and age
        gender_prob, age_pred = age_model.predict(face_gray)

        # Convert predicted indices to integers
        predicted_gender_index = np.argmax(gender_prob)
        predicted_age_index = np.argmax(age_pred)

        # Retrieve gender and age labels
        predicted_gender = "Male" if predicted_gender_index == 0 else "Female"
        predicted_age = predicted_age_index  # Retrieve age range directly from ageList

        print("Gender:", predicted_gender)
        print("Age:", predicted_age)

        # Show the output frame with gender and age labels
        cv2.putText(resultImg, f'{predicted_gender}, {predicted_age}', 
                    (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Encode the image to JPEG format
    ret, encodedImg = cv2.imencode('.jpg', resultImg)
    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['fileToUpload'].read()
        img = Image.open(io.BytesIO(f))
        img_ip = np.asarray(img, dtype="uint8")
        print(img_ip)
        return Response(gen_frames_photo(img_ip,age_model), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
