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
    # Generate dummy detections
    detections = sv.Detections(
        xyxy=np.random.randint(0, 300, (5, 4)).astype(float),
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

@app.route('/analyze-video', methods=['POST'])
def analyze_video():
    try:
        # Get input from form data
        video_url = request.form.get('video_url')
        file = request.files.get('file')
        
        # Generate sample frame for demo
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # Process frame
        annotated_frame, detections = process_frame(frame)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        response = {
            "status": "success",
            "annotated_frame": base64_image,
            "detections": {
                "players": int(np.sum(detections.class_id == PLAYER_ID)),
                "goalkeepers": int(np.sum(detections.class_id == GOALKEEPER_ID)),
                "referees": int(np.sum(detections.class_id == REFEREE_ID)),
                "balls": int(np.sum(detections.class_id == BALL_ID))
            }
        }
        
        return make_response(jsonify(response), 200)
    
    except Exception as e:
        # Log the error for debugging
        print(f"Error in analyze_video: {e}")  
        return make_response(
            jsonify({"status": "error", "detail": str(e)}),
            500
        )

def run_server():
    app.run(host="0.0.0.0", port=9000, debug=False)

if __name__ == '__main__':
    # Start server in background if needed
    # server_thread = threading.Thread(target=run_server, daemon=True)
    # server_thread.start()
    
    # Or run directly
    run_server()
    

