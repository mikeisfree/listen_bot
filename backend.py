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
        self.sample_rate = 16000
        self.connected_clients = set()
        self.current_filename = ""
        self.loop = None
        self.system_task = None
        self.command_task = None
        self.running_sources = set()
        self.transcription_lock = asyncio.Lock()

    async def broadcast(self, message):
        print(f"[LOG] {message.get('text', message.get('message', ''))}")
        if not self.connected_clients:
            return
        data = json.dumps(message)
        dead = set()
        for client in list(self.connected_clients):
            try:
                await client.send(data)
            except websockets.exceptions.ConnectionClosed:
                dead.add(client)
        self.connected_clients -= dead

    def _audio_callback_factory(self, q):
        def callback(indata, frames, time, status):
            if status:
                print(f"Audio Status: {status}", file=sys.stderr)
            q.put(indata.copy())
        return callback

    def _select_device(self, source_type):
        devices = sd.query_devices()
        device_id = None
        if source_type == "system":
            for i, dev in enumerate(devices):
                if "monitor" in dev["name"].lower() and dev["max_input_channels"] > 0:
                    device_id = i
                    break
            if device_id is None:
                for i, dev in enumerate(devices):
                    if any(n in dev["name"].lower() for n in ["pulse", "pipewire"]) and dev["max_input_channels"] > 0:
                        device_id = i
                        break
        else:
            default_input = sd.default.device
            if isinstance(default_input, (list, tuple)) and len(default_input) > 0 and default_input[0] is not None:
                device_id = default_input[0]
            elif isinstance(default_input, int) and default_input >= 0:
                device_id = default_input
            else:
                for i, dev in enumerate(devices):
                    if dev["max_input_channels"] > 0 and "microphone" in dev["name"].lower():
                        device_id = i
                        break
        return device_id, devices

    async def start_transcription(self, model_size, filename, source_type="system", command_mode=False, interval=6.0, device_id=None, shared_model=None):
        self.current_filename = filename
        output_dir = "Transcripts"
        os.makedirs(output_dir, exist_ok=True)

        if command_mode:
            output_path = None
        else:
            output_path = os.path.join(output_dir, f"{filename}.md")
            if not os.path.exists(output_path):
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(f"# Transcription: {filename}\nSource: {source_type}\nDate: {datetime.now()}\n\n---\n\n")

        if shared_model is None:
            await self.broadcast({"type": "status", "text": f"Loading model '{model_size}' for {source_type}..."})

            def load_model():
                try:
                    return WhisperModel(model_size, device="cuda", compute_type="int8")
                except Exception:
                    return WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=4)

            model = await asyncio.get_running_loop().run_in_executor(None, load_model)
            await self.broadcast({"type": "status", "text": f"Model loaded. Source: {source_type}. Recording..."})
        else:
            model = shared_model
            await self.broadcast({"type": "status", "text": f"Ready with shared model. Source: {source_type}. Recording..."})

        if device_id is None:
            device_id, devices = self._select_device(source_type)
        else:
            devices = sd.query_devices()

        device_name = devices[device_id]["name"] if device_id is not None and 0 <= device_id < len(devices) else "Default"
        print(f"[LOG] Using device: {device_name} (ID: {device_id})")

        audio_queue = queue.Queue()
        callback = self._audio_callback_factory(audio_queue)

        if command_mode:
            self.running_sources.add("mic_command")
        else:
            self.running_sources.add(source_type)

        # Trik na PipeWire/PulseAudio - unikalna nazwa klienta PyAudio stream dla rutera Pavucontrol
        # To pozwala GUI Linuxa rozróznic Twoj system stream i mic stream bez laczenia wezlow
        prev_pulse_prop = os.environ.get("PULSE_PROP")
        os.environ["PULSE_PROP"] = f"application.name='listen_bot_{source_type}'"

        audio_buffer = np.array([], dtype=np.float32)
        last_text = ""

        try:
            with sd.InputStream(device=device_id, samplerate=self.sample_rate, channels=1, callback=callback):
                while (source_type == "system" and "system" in self.running_sources) or \
                      (source_type == "mic" and ("mic" in self.running_sources or "mic_command" in self.running_sources)):
                    while not audio_queue.empty():
                        chunk = audio_queue.get()
                        audio_buffer = np.append(audio_buffer, chunk.flatten())

                    if len(audio_buffer) >= self.sample_rate * interval:
                        try:
                            buf_copy = audio_buffer.copy()

                            def run_whisper():
                                segs, _ = model.transcribe(
                                    buf_copy,
                                    beam_size=5,
                                    language="pl",
                                    vad_filter=True,
                                    condition_on_previous_text=False,  # Ogranicza "halucynacje" i zator buforów gdy leci cisza
                                )
                                return [s.text.strip() for s in list(segs) if s.text.strip()]

                            async with self.transcription_lock:
                                found = await asyncio.get_running_loop().run_in_executor(None, run_whisper)
                                
                        except Exception as e:
                            await self.broadcast({"type": "error", "message": f"Transcribe error for {source_type}: {str(e)}"})
                            audio_buffer = np.array([], dtype=np.float32)
                            await asyncio.sleep(0.1)
                            continue

                        if command_mode:
                            recognized = " ".join(found).lower()
                            if "grab" in recognized.split():
                                await self.broadcast({"type": "keyword", "keyword": "grab", "text": "command: grab"})
                                print("[LOG] grab command detected")
                        else:
                            if found:
                                current_text = " ".join(found)
                                await self.broadcast({"type": "transcript", "text": current_text})
                                if output_path:
                                    with open(output_path, "a", encoding="utf-8") as f:
                                        f.write(f"{current_text}  \n")

                        audio_buffer = np.array([], dtype=np.float32)

                    await asyncio.sleep(0.3)

        except asyncio.CancelledError:
            print(f"[LOG] Transcription task cancelled for source: {source_type}")
            raise
        except Exception as e:
            msg = str(e)
            if "Invalid sample rate" in msg or "-9997" in msg:
                 msg += " (Spróbuj wybrać indeks obok 'default', 'pulse' lub 'pipewire'. Bezpośrednie id sprzętowe z ALSA pod Linuksem odrzucają skalowanie do wymaganych natywnie rzędu 16kHz dla Whisper'a!)"
            await self.broadcast({"type": "error", "message": f"Source {source_type} failed: {msg}"})
        finally:
            if prev_pulse_prop is not None:
                os.environ["PULSE_PROP"] = prev_pulse_prop
            else:
                os.environ.pop("PULSE_PROP", None)
            
            key = "mic_command" if command_mode else source_type
            self.running_sources.discard(key)
            if source_type == "system":
                self.system_task = None
            elif source_type == "mic" and command_mode:
                self.command_task = None
            await self.broadcast({"type": "status", "text": f"{source_type.capitalize()} stream stopped."})

    async def ws_handler(self, websocket):
        self.connected_clients.add(websocket)
        try:
            async for message in websocket:
                data = json.loads(message)
                action = data.get("action")
                if action == "start":
                    if self.system_task and not self.system_task.done():
                        self.system_task.cancel()
                    if self.command_task and not self.command_task.done():
                        self.command_task.cancel()

                    source = data.get("source", "system")
                    model_size = data.get("model", "medium")
                    filename = data.get("filename", "web_transcript")
                    interval = float(data.get("interval", 6))

                    if source == "both":
                        await self.broadcast({"type": "status", "text": f"Loading SHARED model '{model_size}' for BOTH streams..."})
                        def load_shared():
                            try:
                                return WhisperModel(model_size, device="cuda", compute_type="int8")
                            except Exception:
                                return WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=4)
                        shared_m = await asyncio.get_running_loop().run_in_executor(None, load_shared)

                        self.system_task = asyncio.create_task(self.start_transcription(model_size, filename, source_type="system", command_mode=False, interval=interval, shared_model=shared_m))
                        await asyncio.sleep(1.0) # Zabezpieczenie przed ucinaniem portu PortAudio ALSA przy dualnym streamie!
                        self.command_task = asyncio.create_task(self.start_transcription(model_size, f"{filename}_cmd", source_type="mic", command_mode=True, interval=interval, shared_model=shared_m))
                        await self.broadcast({"type": "status", "text": "Started both system transcription and microphone listener"})
                    elif source == "system":
                        self.system_task = asyncio.create_task(self.start_transcription(model_size, filename, source_type="system", command_mode=False, interval=interval))
                        await self.broadcast({"type": "status", "text": "Started system transcription"})
                    elif source == "mic":
                        self.command_task = asyncio.create_task(self.start_transcription(model_size, filename, source_type="mic", command_mode=False, interval=interval))
                        await self.broadcast({"type": "status", "text": "Started microphone transcription"})
                    else:
                        await self.broadcast({"type": "error", "message": f"Unknown source '{source}'"})
                elif action == "stop":
                    self.running_sources.clear()
                    if self.system_task and not self.system_task.done():
                        self.system_task.cancel()
                    if self.command_task and not self.command_task.done():
                        self.command_task.cancel()
                    await self.broadcast({"type": "status", "text": "Stopped."})
        finally:
            self.connected_clients.discard(websocket)

    async def cli_input_task(self):
        while True:
            print("\n[CLI] Start transcription? Enter filename (or press Enter to skip): ", end="", flush=True)
            filename = await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)
            filename = filename.strip()
            if filename:
                print("[CLI] Select source: (1) System Audio [Default], (2) Microphone, (3) Both: ", end="", flush=True)
                choice = await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)
                source = "mic" if choice.strip() == "2" else ("both" if choice.strip() == "3" else "system")
                print("[CLI] Select model size (tiny, base, small, medium) [Default: medium]: ", end="", flush=True)
                model_choice = await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)
                model_size = model_choice.strip() or "medium"

                print("[CLI] Enable transcription delay in seconds (default: 6.0): ", end="", flush=True)
                interval_choice = await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)
                try:
                    interval = float(interval_choice.strip())
                except ValueError:
                    interval = 6.0

                # Przechwytywanie ID urządzeń z portaudio
                devices = sd.query_devices()
                sys_device_id = None
                mic_device_id = None

                if source in ["system", "both"]:
                    print("\n[CLI] --- AVAILABLE INPUT DEVICES ---")
                    for i, dev in enumerate(devices):
                        if dev["max_input_channels"] > 0:
                            print(f"  [{i}] {dev['name']}")
                    print("[CLI] Select device ID for SYSTEM AUDIO (or press Enter for auto): ", end="", flush=True)
                    choice = await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)
                    if choice.strip().isdigit():
                        sys_device_id = int(choice.strip())

                if source in ["mic", "both"]:
                    if source != "both":
                        print("\n[CLI] --- AVAILABLE INPUT DEVICES ---")
                        for i, dev in enumerate(devices):
                            if dev["max_input_channels"] > 0:
                                print(f"  [{i}] {dev['name']}")
                    print("[CLI] Select device ID for MICROPHONE (or press Enter for auto): ", end="", flush=True)
                    choice = await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)
                    if choice.strip().isdigit():
                        mic_device_id = int(choice.strip())

                print(f"[CLI] Starting local transcription for: {filename} from {source} (Model: {model_size}, Delay: {interval}s)")

                if source == "both":
                    if self.system_task and not self.system_task.done():
                        self.system_task.cancel()
                    if self.command_task and not self.command_task.done():
                        self.command_task.cancel()
                    
                    print(f"[CLI] Loading SHARED model '{model_size}' ONCE for both streams...")
                    def load_cli():
                        try:
                            return WhisperModel(model_size, device="cuda", compute_type="int8")
                        except Exception:
                            return WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=4)
                    shared_m = await asyncio.get_running_loop().run_in_executor(None, load_cli)
                    
                    self.system_task = asyncio.create_task(self.start_transcription(model_size, filename, source_type="system", command_mode=False, interval=interval, device_id=sys_device_id, shared_model=shared_m))
                    await asyncio.sleep(1.0) # Zabezpieczenie routingu ALSA pcm by nie zawieszać wyboru systemowego Pipewire
                    self.command_task = asyncio.create_task(self.start_transcription(model_size, f"{filename}_cmd", source_type="mic", command_mode=True, interval=interval, device_id=mic_device_id, shared_model=shared_m))
                    await asyncio.gather(self.system_task, self.command_task, return_exceptions=True)
                else:
                    selected_id = sys_device_id if source == "system" else mic_device_id
                    await self.start_transcription(model_size, filename, source_type=source, command_mode=False, interval=interval, device_id=selected_id)
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
