from flask import Flask, request, jsonify
import face_recognition
import requests
from io import BytesIO
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def load_image_from_url(url):
    """Fetches an image from a URL and converts it to a format compatible with face_recognition."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for invalid responses (404, 500, etc.)
        return face_recognition.load_image_file(BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image from {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing image from {url}: {e}")
        return None

def compare_faces(source_url, target_url, threshold):
    """Compares faces between two images and returns match percentage."""
    # Load images
    source_image = load_image_from_url(source_url)
    target_image = load_image_from_url(target_url)

    if source_image is None or target_image is None:
        return {"message": "Error loading images. Please check the image URLs.", "status": 0}

    # Get face encodings and landmarks
    source_face_encodings = face_recognition.face_encodings(source_image)
    target_face_encodings = face_recognition.face_encodings(target_image)

    source_landmarks = face_recognition.face_landmarks(source_image)
    target_landmarks = face_recognition.face_landmarks(target_image)

    # Handle no face detected cases
    if not source_face_encodings:
        return {"message": "No face found in source image!", "match_percentage": 0, "status": 0}
    if not target_face_encodings:
        return {"message": "No face found in target image!", "match_percentage": 0, "status": 0}

    # Use the first detected face encoding
    source_face_encoding = source_face_encodings[0]
    target_face_encoding = target_face_encodings[0]

    # Compare faces
    face_distance = face_recognition.face_distance([source_face_encoding], target_face_encoding)[0]
    match_percentage = round((1.0 - face_distance) * 100, 2)

    # Determine match status based on threshold
    if match_percentage >= threshold:
        return {
            "message": f"Faces match with a percentage of {match_percentage:.2f}%",
            "match_percentage": match_percentage,
            "status": 1,
            "source_landmarks": source_landmarks[0] if source_landmarks else [],
            "target_landmarks": target_landmarks[0] if target_landmarks else []
        }
    else:
        return {
            "message": f"Faces do not match, match percentage is {match_percentage:.2f}%",
            "match_percentage": match_percentage,
            "status": 0
        }

@app.route('/compare-faces-threshold', methods=['POST'])
def compare_faces_api():
    """API endpoint for comparing faces between two image URLs."""
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid JSON request body"}), 400

    source_url = data.get('source_url')
    target_url = data.get('target_url')
    threshold = data.get('threshold', 50)  # Default threshold is 50 if not provided

    if not source_url or not target_url:
        return jsonify({"error": "Both 'source_url' and 'target_url' are required!"}), 400

    result = compare_faces(source_url, target_url, threshold)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8888, debug=True)  # Enable debugging for development
