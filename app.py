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

        # 2. Open video and grab a random frame
        cap = cv2.VideoCapture(selected_video)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return make_response(jsonify({
                "status": "error",
                "detail": "Video has no readable frames"
            }), 400)

        random_frame_index = random.randint(0, frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return make_response(jsonify({
                "status": "error",
                "detail": "Failed to extract frame"
            }), 500)

        # 3. Process frame with your real model + annotators
        
        from inference import get_model
        PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/11"
        PLAYER_DETECTION_MODEL = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key='vmsJ0NWzacPKCysW55Br')

        # frame_generator = sv.get_video_frames_generator(selected_video)
        # frame = next(frame_generator)

        box_annotator = sv.BoxAnnotator(
        color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
        thickness=2
        )
        label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
            text_color=sv.Color.from_hex('#000000')
        )

        result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(detections['class_name'], detections.confidence)
        ]

        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels)

            # Video Game style
        BALL_ID = 0

        ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            thickness=2
        )
        triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex('#FFD700'),
            base=25,
            height=21,
            outline_thickness=1
        )

        result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)

        ball_detections = detections[detections.class_id == BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        all_detections = detections[detections.class_id != BALL_ID]
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections.class_id -= 1

        annotated_frame = frame.copy()
        annotated_frame = ellipse_annotator.annotate(
            scene=annotated_frame,
            detections=all_detections)
        annotated_frame = triangle_annotator.annotate(
            scene=annotated_frame,
            detections=ball_detections)
        
        # Player Tracking
        BALL_ID = 0

        ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            thickness=2
        )
        label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            text_color=sv.Color.from_hex('#000000'),
            text_position=sv.Position.BOTTOM_CENTER
        )
        triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex('#FFD700'),
            base=25,
            height=21,
            outline_thickness=1
        )

        tracker = sv.ByteTrack()
        tracker.reset()

        result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)

        ball_detections = detections[detections.class_id == BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        all_detections = detections[detections.class_id != BALL_ID]
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections.class_id -= 1
        all_detections = tracker.update_with_detections(detections=all_detections)

        labels = [
            f"#{tracker_id}"
            for tracker_id
            in all_detections.tracker_id
        ]

        annotated_frame = frame.copy()
        annotated_frame = ellipse_annotator.annotate(
            scene=annotated_frame,
            detections=all_detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=all_detections,
            labels=labels)
        annotated_frame = triangle_annotator.annotate(
            scene=annotated_frame,
            detections=ball_detections)

        # 4. Encode and respond
        _, buf = cv2.imencode('.jpg', annotated_frame)
        b64 = base64.b64encode(buf).decode('utf-8')
        return make_response(jsonify({
            "status": "success",
            "annotated_frame": b64,
            "detections": {
                "players": int(np.sum(detections.class_id == PLAYER_ID)),
                "goalkeepers": int(np.sum(detections.class_id == GOALKEEPER_ID)),
                "referees": int(np.sum(detections.class_id == REFEREE_ID)),
                "balls": int(np.sum(detections.class_id == BALL_ID))
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
    

