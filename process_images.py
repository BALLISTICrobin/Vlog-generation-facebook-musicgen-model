# --- 1. Imports ---
import os
import shutil
import tempfile
import uuid
import random
import traceback
from typing import List
from contextlib import asynccontextmanager

# Core libraries
import cv2
import requests
import torch
import scipy.io.wavfile
import uvicorn
from PIL import Image
from moviepy import VideoFileClip, AudioFileClip

# FastAPI and Pydantic
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from starlette.responses import FileResponse

# Transformers for music generation
from transformers import pipeline

# --- 2. Global State & Configuration for MusicGen ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_pipeline = None # This will hold the loaded MusicGen model

# --- 3. Lifespan Management (Model Loading/Unloading) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's startup and shutdown events.
    The MusicGen model is loaded on startup to avoid reloading it on every request.
    """
    global model_pipeline
    print("Server starting up...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device for MusicGen: {'CUDA' if device.type == 'cuda' else 'CPU'}")

        print("Loading facebook/musicgen-small model...")
        model_pipeline = pipeline(
            "text-to-audio",
            model="facebook/musicgen-small",
            device=0 if device.type == 'cuda' else -1
        )
        print("MusicGen model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Could not load MusicGen model. Error: {e}")
        traceback.print_exc()

    yield  # The application runs after this point

    print("Server shutting down...")
    if model_pipeline:
        del model_pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Cleaned up resources.")

# --- 4. FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan)

# --- 5. Pydantic Models for API Requests ---
class VideoRequest(BaseModel):
    """Pydantic model to validate the incoming request."""
    image_urls: List[HttpUrl]
    prompt: str

# --- 6. Helper Functions ---
def generate_music(prompt: str, duration: int, file_path: str):
    """Generates music based on a prompt and saves it to a file."""
    global model_pipeline
    if model_pipeline is None:
        raise RuntimeError("Music generation model is not available.")
    
    try:
        print(f"Generating music for prompt: '{prompt}' with duration: {duration}s")
        max_length = int(duration * 50) # Heuristic: 50 tokens per second
        
        random_seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

        music = model_pipeline(prompt, forward_params={"do_sample": True, "max_new_tokens": max_length})

        audio_data = music["audio"][0].T
        scipy.io.wavfile.write(file_path, rate=music["sampling_rate"], data=audio_data)
        print(f"Music saved to {file_path}")
        
    except Exception as e:
        print(f"Error during music generation: {e}")
        traceback.print_exc()
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def process_and_create_video(image_folder: str, music_prompt: str) -> str:
    """
    Core logic: resizes images, creates a silent video, generates music,
    and merges them into a final video file.
    """
    # --- 1. Collect and validate images ---
    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if not image_files:
        raise ValueError("No valid images found in the directory.")
    print(f"Found {len(image_files)} images.")

    # --- 2. Calculate dimensions and video duration ---
    total_width, total_height, readable_images_count = 0, 0, 0
    for file_name in image_files:
        try:
            with Image.open(os.path.join(image_folder, file_name)) as im:
                width, height = im.size
                total_width += width
                total_height += height
                readable_images_count += 1
        except IOError:
            print(f"Warning: Could not read {file_name}, skipping.")
    
    if readable_images_count == 0:
        raise ValueError("None of the image files could be read.")

    mean_width = int(total_width / readable_images_count)
    mean_height = int(total_height / readable_images_count)
    
    # --- Video settings (adjusted for short videos) ---
    FPS = 30
    STILL_DURATION_SEC = 1.5
    TRANSITION_DURATION_SEC = 0.5
    
    video_duration = (len(image_files) * STILL_DURATION_SEC) + \
                     ((len(image_files) - 1) * TRANSITION_DURATION_SEC)
    
    # Enforce 10-second limit
    if video_duration > 10.0:
        raise ValueError("The combination of images and durations exceeds the 10-second limit.")
    print(f"Calculated video duration: {video_duration:.2f}s")


    # --- 3. Resize images ---
    for file_name in image_files:
        file_path = os.path.join(image_folder, file_name)
        with Image.open(file_path) as im:
            im_resized = im.resize((mean_width, mean_height), Image.LANCZOS)
            im_resized.save(file_path, 'JPEG', quality=95)

    # --- 4. Generate silent video with transitions ---
    print("Generating silent video with transitions...")
    silent_video_path = os.path.join(image_folder, 'silent_video.mp4')
    video = cv2.VideoWriter(
        silent_video_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (mean_width, mean_height)
    )
    
    still_frames = int(STILL_DURATION_SEC * FPS)
    transition_frames = int(TRANSITION_DURATION_SEC * FPS)

    for i, file_name in enumerate(image_files):
        current_frame = cv2.imread(os.path.join(image_folder, file_name))
        for _ in range(still_frames):
            video.write(current_frame)
        
        if i < len(image_files) - 1:
            next_frame = cv2.imread(os.path.join(image_folder, image_files[i + 1]))
            for j in range(transition_frames):
                alpha = j / (transition_frames - 1) if transition_frames > 1 else 1.0
                blended = cv2.addWeighted(current_frame, 1 - alpha, next_frame, alpha, 0)
                video.write(blended)
    video.release()
    print("Silent video generated.")

    # --- 5. Generate music ---
    music_path = os.path.join(image_folder, "background_music.wav")
    # Generate slightly longer audio to avoid abrupt cuts
    generate_music(music_prompt, int(video_duration) + 1, music_path)

    # --- 6. Merge video and audio using moviepy ---
    print("Merging video and audio...")
    final_video_path = os.path.join(image_folder, "final_video.mp4")
    
    video_clip = VideoFileClip(silent_video_path)
    audio_clip = AudioFileClip(music_path)
    
    # *** THIS IS THE FIX ***
    # Trim audio to match video duration exactly
    final_audio = audio_clip.subclipped(0, video_clip.duration)
    
    final_clip = video_clip.with_audio(final_audio)
    final_clip.write_videofile(final_video_path, codec='libx264', audio_codec='aac')
    
    # Close clips to release file handles
    audio_clip.close()
    video_clip.close()
    final_clip.close()
    
    print(f"Final video with audio saved to: {final_video_path}")
    return final_video_path

# --- 7. API Endpoint ---
@app.post("/create-video-with-music")
def create_video_from_urls(payload: VideoRequest, background_tasks: BackgroundTasks):
    """
    API endpoint to create a video with generated music from image URLs and a prompt.
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not ready. Please try again later.")
        
    temp_dir = tempfile.mkdtemp()
    background_tasks.add_task(shutil.rmtree, temp_dir) # Cleanup after response

    print(f"Downloading {len(payload.image_urls)} images to {temp_dir}...")
    for i, url in enumerate(payload.image_urls):
        try:
            response = requests.get(str(url), stream=True, timeout=30)
            response.raise_for_status()
            file_extension = os.path.splitext(str(url.path))[-1].lower() or '.jpg'
            if file_extension not in ['.jpg', '.jpeg', '.png']:
                file_extension = '.jpg'
            file_path = os.path.join(temp_dir, f"image_{i:03d}{file_extension}")
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Failed to download image: {url}. Error: {e}")

    try:
        video_path = process_and_create_video(temp_dir, payload.prompt)
        return FileResponse(
            path=video_path,
            media_type='video/mp4',
            filename='generated_video_with_music.mp4'
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

# --- 8. Main execution block ---
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
