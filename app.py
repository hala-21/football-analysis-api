from flask import Flask, request, jsonify, make_response
import supervision as sv
import numpy as np
import cv2
import base64
import threading
import tempfile
import requests
from typing import Optional
import data_loader
import data_processor
import os
import random

app = Flask(__name__)

# Mock models for demo (replace with real models)
class TeamClassifier:
    def predict(self, crops):
        return np.random.randint(0, 2, len(crops))

# Constants
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

# Initialize components once
tracker = sv.ByteTrack()
team_classifier = TeamClassifier()

# Initialize annotators
annotators = {
    'ellipse': sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        thickness=2
    ),
    'label': sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        text_position=sv.Position.BOTTOM_CENTER
    ),
    'triangle': sv.TriangleAnnotator(
        color=sv.Color.from_hex('#FFD700'),
        base=25,
        height=21,
        outline_thickness=1
    )
}

def process_frame(frame: np.ndarray):
    """Mock frame processing for demo"""
    # Generate valid dummy bounding boxes
    xyxy = np.random.randint(0, 300, (5, 4)).astype(float)
    xyxy[:, [0, 2]] = np.sort(xyxy[:, [0, 2]], axis=1)  # Ensure x1 < x2
    xyxy[:, [1, 3]] = np.sort(xyxy[:, [1, 3]], axis=1)  # Ensure y1 < y2

    detections = sv.Detections(
        xyxy=xyxy,
        confidence=np.random.rand(5),
        class_id=np.random.randint(0, 4, 5)
    )
    
    # Annotate frame
    annotated = frame.copy()
    annotated = annotators['ellipse'].annotate(annotated, detections)
    return annotated, detections


@app.route('/api/v1/teams', methods=['GET'])
def get_teams():
    teams_data = data_loader.load_teams_data()
    return jsonify(teams_data)

@app.route('/api/v1/players', methods=['GET'])
def get_players():
    team_name = request.args.get('team_name')
    players_data = data_loader.load_players_data(team_name)
    return jsonify(players_data)

@app.route('/api/v1/player-stats', methods=['GET'])
def get_player_stats():
    player_name = request.args.get('player_name')
    stats_data = data_processor.process_player_stats(player_name)
    return jsonify(stats_data)

@app.route('/api/v1/team-stats', methods=['GET'])
def get_team_stats():
    team_name = request.args.get('team_name')
    stats_data = data_processor.process_team_stats(team_name)
    return jsonify(stats_data)

import os
import random

@app.route('/analyze-video', methods=['POST', 'GET'])
def analyze_video():
    try:
        # 1. Find the first video in uploads/
        uploads_dir = 'uploads'
        video_files = [
            f for f in os.listdir(uploads_dir)
            if f.lower().endswith(('.mp4', '.avi', '.mov'))
        ]
        if not video_files:
            return make_response(jsonify({
                "status": "error",
                "detail": "No video files found in uploads/"
            }), 404)

        selected_video = os.path.join(uploads_dir, video_files[0])

        # 2. Open video and check its length
        cap = cv2.VideoCapture(selected_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = frame_count / fps

        if frame_count <= 0 or fps <= 0:
            cap.release()
            return make_response(jsonify({
                "status": "error",
                "detail": "Video has no readable frames or invalid FPS"
            }), 400)

        # Determine the number of frames to process
        max_duration = 60  # 1 minute in seconds
        frames_to_process = int(min(video_duration, max_duration) * fps)

        # Skip every nth frame to speed up processing
        frame_skip = 5  # Process every 5th frame
        frames_to_process = frames_to_process // frame_skip

        # Initialize annotators and tracker
        box_annotator = sv.BoxAnnotator(
            color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
            thickness=2
        )
        label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
            text_color=sv.Color.from_hex('#000000')
        )
        tracker = sv.ByteTrack()
        tracker.reset()

        # Load the model
        from inference import get_model
        PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/11"
        PLAYER_DETECTION_MODEL = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key='vmsJ0NWzacPKCysW55Br')

        # Process the video frame by frame
        annotated_frames = []
        for frame_idx in range(frames_to_process):
            # Skip frames
            for _ in range(frame_skip - 1):
                cap.grab()  # Skip the next frame

            ret, frame = cap.read()
            if not ret:
                break

            # Apply the model to the frame
            result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
            detections = sv.Detections.from_inference(result)

            # Annotate the frame
            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence
                in zip(detections['class_name'], detections.confidence)
            ]

            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels
            )

            # Track players
            all_detections = detections.with_nms(threshold=0.5, class_agnostic=True)
            all_detections = tracker.update_with_detections(detections=all_detections)

            annotated_frames.append(annotated_frame)

        cap.release()

        # Encode the annotated frames into a video
        temp_video_path = os.path.join(tempfile.gettempdir(), 'annotated_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width, _ = annotated_frames[0].shape
        out = cv2.VideoWriter(temp_video_path, fourcc, fps // frame_skip, (width, height))

        for frame in annotated_frames:
            out.write(frame)
        out.release()

        # Encode the video to base64
        with open(temp_video_path, 'rb') as f:
            video_data = f.read()
        b64_video = base64.b64encode(video_data).decode('utf-8')

        return make_response(jsonify({
            "status": "success",
            "annotated_video": b64_video,
            "detections_summary": {
                "total_frames_processed": len(annotated_frames),
                "video_duration_processed": min(video_duration, max_duration)
            }
        }), 200)

    except Exception as e:
        app.logger.exception("Error in analyze_video")
        return make_response(jsonify({
            "status": "error",
            "detail": str(e)
        }), 500)
    
@app.route('/')
def index():
    return app.send_static_file('index.html')

def run_server():
    app.run(host="0.0.0.0", port=9000, debug=False)

if __name__ == '__main__':
    # Start server in background if needed
    # server_thread = threading.Thread(target=run_server, daemon=True)
    # server_thread.start()
    
    # Or run directly
    run_server()
    

