from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import os
import scipy.io.wavfile
import torch
from transformers import pipeline
import random
import traceback
import uuid
from contextlib import asynccontextmanager

# --- Global State & Configuration ---

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global variable to hold the model pipeline
model_pipeline = None

# --- Lifespan Management (for model loading/unloading) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs ONCE when the server starts up.
    global model_pipeline
    print("Server starting up...")
    try:
        # Set device (GPU or CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {'CUDA' if device.type == 'cuda' else 'CPU'}")

        # Load the SMALLER MusicGen model into the global variable
        # This is the key change to prevent out-of-memory errors on smaller GPUs
        print("Loading facebook/musicgen-small model... (this may take a moment)")
        model_pipeline = pipeline(
            "text-to-audio",
            model="facebook/musicgen-small",  # <-- Using the small model
            device=0 if device.type == 'cuda' else -1
        )
        print("Model loaded successfully.")

    except Exception as e:
        print(f"FATAL: Could not load model. Error: {e}")
        traceback.print_exc()

    yield # The application runs after this point

    # This code runs ONCE when the server shuts down.
    print("Server shutting down...")
    if model_pipeline:
        del model_pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Cleaned up resources.")

# --- FastAPI App Initialization ---

app = FastAPI(lifespan=lifespan)

class MusicRequest(BaseModel):
    prompt: str
    duration: int  # Duration for each track in seconds

# --- Helper Function for Background Task ---

def generate_and_save_music(prompt: str, duration: int, file_path1: str, file_path2: str):
    """
    This is the core, long-running function that will be executed in the background.
    """
    global model_pipeline
    if model_pipeline is None:
        print("Error: Model pipeline is not available.")
        return

    try:
        print(f"Starting music generation for prompt: '{prompt}'")
        
        # The model generates audio based on a length parameter.
        # For MusicGen, a common heuristic is that it generates at 50 tokens/sec.
        max_length = duration * 50

        # Generate first track with a random seed
        random_seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            
        music1 = model_pipeline(prompt, forward_params={"do_sample": True, "max_new_tokens": max_length})

        # Generate second track with a different seed to ensure variation
        torch.manual_seed(random_seed + 1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed + 1)

        music2 = model_pipeline(prompt, forward_params={"do_sample": True, "max_new_tokens": max_length})

        # Save audio files
        audio_data1 = music1["audio"][0].T
        scipy.io.wavfile.write(file_path1, rate=music1["sampling_rate"], data=audio_data1)
        print(f"Saved first track to {file_path1}")

        audio_data2 = music2["audio"][0].T
        scipy.io.wavfile.write(file_path2, rate=music2["sampling_rate"], data=audio_data2)
        print(f"Saved second track to {file_path2}")

    except Exception as e:
        print(f"Error during music generation: {e}")
        traceback.print_exc()
    finally:
        # Clean up GPU memory after generation if needed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Finished music generation task.")


# --- API Endpoint ---

@app.post("/generate-music/")
async def generate_music_endpoint(request: MusicRequest, background_tasks: BackgroundTasks):
    if request.duration <= 0:
        raise HTTPException(status_code=400, detail="Duration must be greater than zero")
    
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or ready. Please try again later.")

    # Generate unique filenames to avoid conflicts
    task_id = str(uuid.uuid4())
    output_dir = os.path.join(os.getcwd(), "generated_music")
    os.makedirs(output_dir, exist_ok=True) # Ensure the directory exists
    
    output1 = os.path.join(output_dir, f"{task_id}_song1.wav")
    output2 = os.path.join(output_dir, f"{task_id}_song2.wav")

    # Add the long-running task to the background
    background_tasks.add_task(
        generate_and_save_music,
        request.prompt,
        request.duration,
        output1,
        output2
    )

    # Immediately return a response to the client
    return {
        "message": "Music generation started in the background.",
        "task_id": task_id,
        "status_info": "The files will be saved on the server in the 'generated_music' folder.",
        "expected_files": {
            "song1": output1,
            "song2": output2
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
