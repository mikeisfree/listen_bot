import os
import sys
import ctypes

# --- NVIDIA LIBRARY PATH HACK ---
# We use ctypes to explicitly load the libraries so the process has them in memory.
def setup_cuda_paths():
    venv_base = os.getcwd() # Use current working directory as base
    site_packages = os.path.join(venv_base, ".venv", "lib", "python3.13", "site-packages")
    
    # Common shared library paths within the venv
    potential_libs = [
        os.path.join(site_packages, "nvidia", "cublas", "lib", "libcublas.so.12"),
        os.path.join(site_packages, "nvidia", "cudnn", "lib", "libcudnn.so.9"),
        os.path.join(site_packages, "nvidia", "cublas", "lib", "libcublasLt.so.12"),
    ]
    
    # Also add them to the environment search path
    lib_dirs = [os.path.dirname(p) for p in potential_libs if os.path.exists(p)]
    if lib_dirs:
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + (":" + current_ld if current_ld else "")
        
        # Force load them into the process memory
        for lib_path in potential_libs:
            if os.path.exists(lib_path):
                try:
                    ctypes.CDLL(lib_path)
                    # print(f"[DEBUG] Loaded {os.path.basename(lib_path)}")
                except Exception as e:
                    pass
        print(f"[INFO] CUDA libraries found and mapped from .venv")

setup_cuda_paths()
# -------------------------------

import asyncio
import sounddevice as sd
import numpy as np
import queue
import websockets
import json
from datetime import datetime
from faster_whisper import WhisperModel

# CONFIGURATION
MODEL_SIZE = "medium"
LANGUAGE = "pl"
VAD_FILTER = True

class TranscriptionServer:
    def __init__(self, filename):
        print(f"--- Initializing Whisper model '{MODEL_SIZE}' ---")
        try:
            # We use int8 for GTX 1060 (Pascal architecture)
            print("Action: Attempting GPU (CUDA) acceleration with int8...")
            self.model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="int8")
            print("[SUCCESS] GPU Acceleration enabled via int8!")
        except Exception as e:
            print(f"[WARN] GPU Initialization failed: {e}")
            print("Action: Falling back to CPU...")
            self.model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=4)

        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        self.output_path = os.path.join("Transcripts", f"{filename}.md")
        
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio Status: {status}", file=sys.stderr)
        self.audio_queue.put(indata.copy())

    async def process_audio(self):
        print(f"\n[ACTIVE] Logging to: '{self.output_path}'")
        print("[STATUS] Recording... (Ctrl+C to stop)\n")
        
        initial_prompt = "To jest profesjonalna transkrypcja rozmowy dotyczącej technologii, prawa i sztucznej inteligencji."
        last_text = ""

        if not os.path.exists(self.output_path):
            with open(self.output_path, "w", encoding="utf-8") as f:
                f.write(f"# Transcription: {os.path.basename(self.output_path)}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n")

        audio_buffer = np.array([], dtype=np.float32)
        
        while True:
            try:
                while not self.audio_queue.empty():
                    chunk = self.audio_queue.get()
                    audio_buffer = np.append(audio_buffer, chunk.flatten())

                if len(audio_buffer) >= self.sample_rate * 5:
                    segments, info = self.model.transcribe(
                        audio_buffer, 
                        beam_size=5, 
                        language=LANGUAGE,
                        vad_filter=VAD_FILTER,
                        initial_prompt=f"{initial_prompt} {last_text}"[-200:], 
                        condition_on_previous_text=True
                    )
                    
                    new_segments = []
                    for segment in segments:
                        text = segment.text.strip()
                        if text and text not in last_text:
                            new_segments.append(text)

                    if new_segments:
                        current_text = " ".join(new_segments)
                        with open(self.output_path, "a", encoding="utf-8") as f:
                            f.write(f"{current_text}  \n")
                            f.flush()
                        last_text = current_text
                    
                    audio_buffer = np.array([], dtype=np.float32)
                
                await asyncio.sleep(0.4)
            except Exception as e:
                print(f"\n[FATAL ERROR] Transcription session failed: {e}")
                break

    async def start(self):
        server = await websockets.serve(lambda w, p: asyncio.sleep(3600), "localhost", 8765)
        
        devices = sd.query_devices()
        device_id = None
        for i, dev in enumerate(devices):
            if any(name in dev['name'].lower() for name in ['pulse', 'pipewire']) and dev['max_input_channels'] > 0:
                device_id = i
                break
        
        if device_id is None:
            device_id = sd.default.device[0]

        asyncio.create_task(self.process_audio())

        with sd.InputStream(device=device_id, samplerate=self.sample_rate, channels=1, callback=self.audio_callback):
            await server.wait_closed()

if __name__ == "__main__":
    os.makedirs("Transcripts", exist_ok=True)
    f_name = input("Enter transcript filename (without .md): ").strip()
    if not f_name:
        f_name = f"transcript_{datetime.now().strftime('%M%S')}"
    
    app = TranscriptionServer(f_name)
    try:
        asyncio.run(app.start())
    except KeyboardInterrupt:
        print(f"\nSaved results in: Transcripts/{f_name}.md")
