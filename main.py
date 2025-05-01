from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
import shutil
from dotenv import load_dotenv
from analyzer import VideoAnalyzer

load_dotenv()

app = FastAPI()
analyzer = VideoAnalyzer()

@app.post("/analyze-video")
async def analyze_video(file: UploadFile):
    # Ensure upload directory exists
    upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save uploaded file
    temp_path = os.path.join(upload_dir, file.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Process video
        result = analyzer.process_video(temp_path, upload_dir)
        
        # Return the processed video
        return FileResponse(
            result["output_path"],
            media_type="video/mp4",
            filename=os.path.basename(result["output_path"])
        )
        
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    finally:
        # Cleanup - remove original upload
        if os.path.exists(temp_path):
            os.remove(temp_path)