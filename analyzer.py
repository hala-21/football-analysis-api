import cv2
import os
from datetime import datetime
from sklearn.cluster import KMeans
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()

class VideoAnalyzer:
    def __init__(self):
        rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
        project = rf.workspace().project("football-field-detection-f07vi")
        self.model = project.version(14).model

    def classify_team_colors(self, detections):
        jersey_colors = []
        player_indices = []
        
        for i, pred in enumerate(detections):
            if pred['class'] == 'player' and 'color' in pred:
                jersey_colors.append([
                    pred['color']['r'],
                    pred['color']['g'],
                    pred['color']['b']
                ])
                player_indices.append(i)

        labels = [-1] * len(detections)
        if len(jersey_colors) >= 2:
            kmeans = KMeans(n_clusters=2, random_state=0).fit(jersey_colors)
            for idx, player_idx in enumerate(player_indices):
                labels[player_idx] = int(kmeans.labels_[idx])

        return labels

    def process_video(self, input_path: str, output_dir: str) -> str:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_path = os.path.join(output_dir, f"result_{timestamp}.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Predict with Roboflow model
            predictions = self.model.predict(frame_rgb, confidence=40).json()
            detections = predictions['predictions']
            
            # Classify teams
            team_labels = self.classify_team_colors(detections)

            # Draw annotations
            for i, pred in enumerate(detections):
                x = int(pred['x'] - pred['width']/2)
                y = int(pred['y'] - pred['height']/2)
                w = int(pred['width'])
                h = int(pred['height'])
                
                color = (0, 255, 0) if team_labels[i] == 0 else (255, 0, 0)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(
                    frame, 
                    f"{pred['class']} {pred['confidence']:.2f} T{team_labels[i]}", 
                    (x, y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    color, 
                    2
                )

            out.write(frame)

        cap.release()
        out.release()
        
        return output_path