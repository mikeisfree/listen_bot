import os
import sys
import ctypes
import asyncio
import sounddevice as sd
import numpy as np
import queue
import websockets
import json
from faster_whisper import WhisperModel

# ─── CUDA PATH SETUP ─────────────────────────────────────────────────────────
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
                try:
                    ctypes.CDLL(lib_path)
                except Exception:
                    pass

setup_cuda_paths()

# ─── SHARED MODEL REGISTRY ────────────────────────────────────────────────────
# Models are loaded once and shared across all client sessions to save memory.
# Inference calls are serialized per model via asyncio locks.
_models: dict[str, WhisperModel] = {}
_infer_locks: dict[str, asyncio.Lock] = {}
_load_lock: asyncio.Lock | None = None


def _get_load_lock() -> asyncio.Lock:
    global _load_lock
    if _load_lock is None:
        _load_lock = asyncio.Lock()
    return _load_lock


async def get_model(model_size: str) -> tuple[WhisperModel, asyncio.Lock]:
    if model_size not in _models:
        async with _get_load_lock():
            if model_size not in _models:
                def _load():
                    try:
                        return WhisperModel(model_size, device="cuda", compute_type="int8")
                    except Exception:
                        return WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=4)
                _models[model_size] = await asyncio.get_running_loop().run_in_executor(None, _load)
                _infer_locks[model_size] = asyncio.Lock()
    return _models[model_size], _infer_locks[model_size]


# ─── PER-CLIENT SESSION ───────────────────────────────────────────────────────
class ClientSession:
    """Fully isolated state for one connected browser client."""

    SAMPLE_RATE = 16000

    def __init__(self, websocket):
        self.ws = websocket
        self.system_task: asyncio.Task | None = None
        self.command_task: asyncio.Task | None = None
        self.running_sources: set[str] = set()

    async def send(self, msg: dict):
        try:
            await self.ws.send(json.dumps(msg))
        except websockets.exceptions.ConnectionClosed:
            pass

    def _audio_callback_factory(self, q: queue.Queue):
        def callback(indata, frames, time, status):
            if status:
                print(f"Audio Status: {status}", file=sys.stderr)
            q.put(indata.copy())
        return callback

    def _select_device(self, source_type: str):
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

    async def start_transcription(
        self,
        model_size: str,
        source_type: str,
        language: str,
        interval: float,
        command_mode: bool = False,
        device_id: int | None = None,
    ):
        await self.send({"type": "status", "text": f"Loading model '{model_size}'..."})
        model, infer_lock = await get_model(model_size)
        await self.send({"type": "status", "text": f"Ready. Source: {source_type}. Recording..."})

        if device_id is None:
            device_id, devices = self._select_device(source_type)
        else:
            devices = sd.query_devices()

        device_name = (
            devices[device_id]["name"]
            if device_id is not None and 0 <= device_id < len(devices)
            else "Default"
        )
        print(f"[LOG] [{id(self)}] {source_type} → device: {device_name} (ID: {device_id})")

        audio_queue: queue.Queue = queue.Queue()
        callback = self._audio_callback_factory(audio_queue)

        source_key = "mic_command" if command_mode else source_type
        self.running_sources.add(source_key)

        prev_pulse_prop = os.environ.get("PULSE_PROP")
        os.environ["PULSE_PROP"] = f"application.name='listen_bot_{source_type}_{id(self)}'"

        audio_buffer = np.array([], dtype=np.float32)

        try:
            with sd.InputStream(device=device_id, samplerate=self.SAMPLE_RATE, channels=1, callback=callback):
                while (
                    (source_type == "system" and "system" in self.running_sources)
                    or (source_type == "mic" and ("mic" in self.running_sources or "mic_command" in self.running_sources))
                ):
                    while not audio_queue.empty():
                        chunk = audio_queue.get()
                        audio_buffer = np.append(audio_buffer, chunk.flatten())

                    if len(audio_buffer) >= self.SAMPLE_RATE * interval:
                        try:
                            buf_copy = audio_buffer.copy()

                            def run_whisper():
                                segs, _ = model.transcribe(
                                    buf_copy,
                                    beam_size=5,
                                    language=language,
                                    vad_filter=True,
                                    condition_on_previous_text=False,
                                )
                                return [s.text.strip() for s in list(segs) if s.text.strip()]

                            async with infer_lock:
                                found = await asyncio.get_running_loop().run_in_executor(None, run_whisper)

                        except Exception as e:
                            await self.send({"type": "error", "message": f"Transcribe error: {e}"})
                            audio_buffer = np.array([], dtype=np.float32)
                            await asyncio.sleep(0.1)
                            continue

                        if command_mode:
                            recognized = " ".join(found).lower()
                            if "grab" in recognized.split():
                                await self.send({"type": "keyword", "keyword": "grab", "text": "command: grab"})
                                print(f"[LOG] [{id(self)}] grab command detected")
                        else:
                            if found:
                                await self.send({"type": "transcript", "text": " ".join(found)})

                        audio_buffer = np.array([], dtype=np.float32)

                    await asyncio.sleep(0.3)

        except asyncio.CancelledError:
            print(f"[LOG] [{id(self)}] Task cancelled: {source_type}")
            raise
        except Exception as e:
            msg = str(e)
            if "Invalid sample rate" in msg or "-9997" in msg:
                msg += " (Try 'pulse' or 'pipewire' device index — direct ALSA hw IDs reject 16kHz resampling required by Whisper.)"
            await self.send({"type": "error", "message": f"Source {source_type} failed: {msg}"})
        finally:
            if prev_pulse_prop is not None:
                os.environ["PULSE_PROP"] = prev_pulse_prop
            else:
                os.environ.pop("PULSE_PROP", None)
            self.running_sources.discard(source_key)
            if source_type == "system":
                self.system_task = None
            elif source_type == "mic":
                self.command_task = None
            await self.send({"type": "status", "text": f"{source_type.capitalize()} stream stopped."})

    async def _cancel_tasks(self):
        self.running_sources.clear()
        for task in [self.system_task, self.command_task]:
            if task and not task.done():
                task.cancel()
        self.system_task = None
        self.command_task = None

    async def handle(self):
        try:
            async for raw in self.ws:
                data = json.loads(raw)
                action = data.get("action")

                if action == "start":
                    await self._cancel_tasks()
                    source = data.get("source", "system")
                    model_size = data.get("model", "medium")
                    language = data.get("language", "pl")
                    interval = float(data.get("interval", 6))

                    if source == "both":
                        # Pre-warm model so both streams share it from cache
                        await self.send({"type": "status", "text": f"Loading shared model '{model_size}'..."})
                        await get_model(model_size)
                        self.system_task = asyncio.create_task(
                            self.start_transcription(model_size, "system", language, interval)
                        )
                        await asyncio.sleep(1.0)  # avoid PortAudio port collision on dual stream
                        self.command_task = asyncio.create_task(
                            self.start_transcription(model_size, "mic", language, interval, command_mode=True)
                        )
                        await self.send({"type": "status", "text": "Started both streams."})

                    elif source == "system":
                        self.system_task = asyncio.create_task(
                            self.start_transcription(model_size, "system", language, interval)
                        )

                    elif source == "mic":
                        self.command_task = asyncio.create_task(
                            self.start_transcription(model_size, "mic", language, interval)
                        )

                    else:
                        await self.send({"type": "error", "message": f"Unknown source '{source}'"})

                elif action == "stop":
                    await self._cancel_tasks()
                    await self.send({"type": "status", "text": "Stopped."})

        finally:
            await self._cancel_tasks()


# ─── ENTRYPOINT ───────────────────────────────────────────────────────────────
async def ws_handler(websocket):
    session = ClientSession(websocket)
    await session.handle()


async def main():
    port = int(os.environ.get("PORT", 8765))
    async with websockets.serve(ws_handler, "0.0.0.0", port):
        print(f"[LOG] ListenBot backend — ws://0.0.0.0:{port}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
