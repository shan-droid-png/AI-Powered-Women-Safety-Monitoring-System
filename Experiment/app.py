from flask import Flask, request, jsonify, render_template, send_file
import os
from datetime import datetime

app = Flask(__name__)

# Path to the folder where videos will be saved
VIDEO_FOLDER = os.path.join(os.getcwd(), 'videos')
os.makedirs(VIDEO_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist

@app.route('/')
def index():
    return render_template('index.html')  # Serve the HTML file from `templates/`

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"message": "No video file provided!"}), 400

    video = request.files['video']

    # Generate a unique filename using a timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f"{timestamp}_{video.filename}"
    save_path = os.path.join(VIDEO_FOLDER, filename)

    # Save video to the 'videos' folder
    video.save(save_path)

    return jsonify({"message": f"Video '{filename}' uploaded and saved successfully!"}), 200

@app.route('/videos/<filename>', methods=['GET'])
def get_video(filename):
    file_path = os.path.join(VIDEO_FOLDER, filename)

    if not os.path.exists(file_path):
        return jsonify({"message": "Video not found!"}), 404

    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
