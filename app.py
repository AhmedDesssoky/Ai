from flask import Flask, request, jsonify
from pathlib import Path
from flask import Flask
import pickle
from flask_cors import CORS
import face_recognition
from collections import Counter
import logging, ngrok
import subprocess
logging.basicConfig(level=logging.INFO)
tunnel = ngrok.werkzeug_develop()





app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)


def encode_known_faces(model="hog", encodings_location=DEFAULT_ENCODINGS_PATH):
    names = []
    encodings = []

    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)


def recognize_faces(image_location, model="hog", encodings_location=DEFAULT_ENCODINGS_PATH):
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    input_face_locations = face_recognition.face_locations(input_image, model=model)
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

    recognized_faces = []

    for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        recognized_faces.append({"id": name})

    return recognized_faces


def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding)
    votes = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match)
    if votes:
        return votes.most_common(1)[0][0]



@app.route("/recognize_faces", methods=["POST"])
def recognize_faces_endpoint():
    if "image" not in request.files:
        return jsonify({"message": "No image file provided."}), 400

    image = request.files["image"]
    save_path = Path("validation") / image.filename
    image.save(save_path)

    recognized_faces = recognize_faces(save_path)

    return jsonify(recognized_faces),200

###api2

@app.route("/add_face", methods=["POST"])
def add_face():
    if "id" not in request.form:
        return jsonify({"message": "Name parameter is missing."}), 400

    name = request.form["id"]
    image_files = request.files.getlist("image")

    if len(image_files) != 3:
        return jsonify({"message": "Three image files are required."}), 400

    # Create folder with the person's name
    person_folder = Path("training") / name
    person_folder.mkdir(parents=True, exist_ok=True)

    # Save the images with the person's name
    for image_file in image_files:
        image_path = person_folder / image_file.filename
        image_file.save(image_path)

    # Re-encode known faces
    encode_known_faces()

    return jsonify({"message": "Faces added successfully."}),200



if __name__ == "__main__":
    app.run(debug=True)

# Start Ngrok tunnel
ngrok_process = subprocess.Popen(['ngrok', 'http', '5000'])

try:
    app.run(debug=True)
finally:
    # Terminate Ngrok tunnel when Flask app is stopped
    ngrok_process.terminate()

