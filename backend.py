import os
import sys
import ctypes
import asyncio
import sounddevice as sd
import numpy as np
import queue
import websockets
import json
from datetime import datetime
from faster_whisper import WhisperModel

# --- NVIDIA LIBRARY PATH HACK ---
def setup_cuda_paths():
    venv_base = os.getcwd()
    site_packages = os.path.join(venv_base, ".venv", "lib", "python3.13", "site-packages")
    potential_libs = [
        os.path.join(site_packages, "nvidia", "cublas", "lib", "libcublas.so.12"),
        os.path.join(site_packages, "nvidia", "cudnn", "lib", "libcudnn.so.9"),
        os.path.join(site_packages, "nvidia", "cublas", "lib", "libcublasLt.so.12"),
    ]
    lib_dirs = [os.path.dirname(p) for p in potential_libs if os.path.exists(p)]
    if lib_dirs:
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + (":" + current_ld if current_ld else "")
        for lib_path in potential_libs:
            if os.path.exists(lib_path):
                try: ctypes.CDLL(lib_path)
                except: pass
setup_cuda_paths()
# -------------------------------

class TranscriptionManager:
    def __init__(self):
        self.model = None
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        self.is_running = False
        self.connected_clients = set()
        self.current_filename = ""
        self.loop = None

    async def broadcast(self, message):
        print(f"[LOG] {message.get('text', message.get('message', ''))}")
        if not self.connected_clients: return
        data = json.dumps(message)
        await asyncio.gather(*[client.send(data) for client in self.connected_clients])

    def audio_callback(self, indata, frames, time, status):
        if status: print(f"Audio Status: {status}", file=sys.stderr)
        if self.is_running: self.audio_queue.put(indata.copy())

    async def start_transcription(self, model_size, filename, source_type="system"):
        if self.is_running: return
        try:
            self.current_filename = filename
            output_dir = "Transcripts"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{filename}.md")

            await self.broadcast({"type": "status", "text": f"Loading model '{model_size}'..."})
            
            def load():
                try:
                    return WhisperModel(model_size, device="cuda", compute_type="int8")
                except:
                    return WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=4)
            
            self.model = await asyncio.get_running_loop().run_in_executor(None, load)
            await self.broadcast({"type": "status", "text": f"Model loaded. Source: {source_type}. Recording..."})

            if not os.path.exists(output_path):
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(f"# Transcription: {filename}\nSource: {source_type}\nDate: {datetime.now()}\n\n---\n\n")

            self.is_running = True
            audio_buffer = np.array([], dtype=np.float32)
            last_text = ""
            
            # Select Audio Device
            devices = sd.query_devices()
            device_id = None
            
            if source_type == "system":
                for i, dev in enumerate(devices):
                    if 'monitor' in dev['name'].lower() and dev['max_input_channels'] > 0:
                        device_id = i; break
                if device_id is None:
                    # Fallback to pulse/pipewire if generic monitor not found
                    for i, dev in enumerate(devices):
                        if any(n in dev['name'].lower() for n in ['pulse', 'pipewire']) and dev['max_input_channels'] > 0:
                            device_id = i; break
            else: # source_type == "mic"
                device_id = sd.default.device[0] # Default System Input
            
            device_name = devices[device_id]['name'] if device_id is not None else "Default"
            print(f"[LOG] Using device: {device_name} (ID: {device_id})")

            with sd.InputStream(device=device_id, samplerate=self.sample_rate, channels=1, callback=self.audio_callback):
                while self.is_running:
                    while not self.audio_queue.empty():
                        chunk = self.audio_queue.get()
                        audio_buffer = np.append(audio_buffer, chunk.flatten())

                    if len(audio_buffer) >= self.sample_rate * 4:
                        segments, _ = self.model.transcribe(
                            audio_buffer, beam_size=5, language="pl", 
                            vad_filter=True, condition_on_previous_text=True
                        )
                        
                        found = []
                        for s in segments:
                            t = s.text.strip()
                            if t and t not in last_text:
                                found.append(t)
                                last_text = t # Update last_text per segment

                        if found:
                            current_text = " ".join(found)
                            await self.broadcast({"type": "transcript", "text": current_text})
                            with open(output_path, "a", encoding="utf-8") as f:
                                f.write(f"{current_text}  \n"); f.flush()
                        
                        audio_buffer = np.array([], dtype=np.float32)
                    await asyncio.sleep(0.4)

        except Exception as e:
            await self.broadcast({"type": "error", "message": f"Source {source_type} failed: {str(e)}"})
            self.is_running = False

    async def ws_handler(self, websocket):
        self.connected_clients.add(websocket)
        try:
            async for message in websocket:
                data = json.loads(message)
                action = data.get("action")
                if action == "start":
                    asyncio.create_task(self.start_transcription(
                        data.get("model", "medium"), 
                        data.get("filename", "web_transcript"),
                        data.get("source", "system")
                    ))
                elif action == "stop":
                    self.is_running = False
        finally:
            self.connected_clients.remove(websocket)

    async def cli_input_task(self):
        """Asynchronously waits for CLI input to start transcription without Web."""
        while True:
            print("\n[CLI] Start transcription? Enter filename (or press Enter to skip): ", end="", flush=True)
            filename = await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)
            filename = filename.strip()
            if filename:
                print("[CLI] Select source: (1) System Audio [Default], (2) Microphone: ", end="", flush=True)
                choice = await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)
                source = "mic" if choice.strip() == "2" else "system"
                
                print(f"[CLI] Starting local transcription for: {filename} from {source}")
                await self.start_transcription("medium", filename, source)
            await asyncio.sleep(1)

async def main():
    manager = TranscriptionManager()
    manager.loop = asyncio.get_running_loop()
    
    # Start WebSocket server
    ws_server = await websockets.serve(manager.ws_handler, "localhost", 8765)
    print("--- Backend Listening BOT ---")
    print("WebSocket: ws://localhost:8765")
    
    # Start CLI input task
    asyncio.create_task(manager.cli_input_task())
    
    await ws_server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
