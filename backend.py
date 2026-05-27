import os
import ctypes
import asyncio
import numpy as np
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
    """Fully isolated state for one connected browser client.

    Audio arrives as raw Float32 PCM (16 kHz, mono) binary WebSocket frames.
    Text frames carry JSON control messages: {"action": "start"|"stop", ...}
    """

    SAMPLE_RATE = 16000

    def __init__(self, websocket):
        self.ws = websocket
        self.model: WhisperModel | None = None
        self.infer_lock: asyncio.Lock | None = None
        self.model_size = "medium"
        self.language = "pl"
        self.interval = 6.0
        self.running = False
        self.audio_buffer = np.array([], dtype=np.float32)

    async def send(self, msg: dict):
        try:
            await self.ws.send(json.dumps(msg))
        except websockets.exceptions.ConnectionClosed:
            pass

    async def _transcribe_buffer(self):
        if len(self.audio_buffer) < self.SAMPLE_RATE * 0.5:
            self.audio_buffer = np.array([], dtype=np.float32)
            return

        buf_copy = self.audio_buffer.copy()
        self.audio_buffer = np.array([], dtype=np.float32)

        model = self.model
        infer_lock = self.infer_lock
        language = self.language

        def run_whisper():
            segs, _ = model.transcribe(
                buf_copy,
                beam_size=5,
                language=language,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            return [s.text.strip() for s in list(segs) if s.text.strip()]

        try:
            async with infer_lock:
                found = await asyncio.get_running_loop().run_in_executor(None, run_whisper)

            if found:
                text = " ".join(found)
                await self.send({"type": "transcript", "text": text})
                if "grab" in text.lower().split():
                    await self.send({"type": "keyword", "keyword": "grab", "text": "command: grab"})
        except Exception as e:
            await self.send({"type": "error", "message": f"Transcribe error: {e}"})

    async def handle(self):
        try:
            async for message in self.ws:
                if isinstance(message, str):
                    data = json.loads(message)
                    action = data.get("action")

                    if action == "start":
                        self.running = False
                        self.audio_buffer = np.array([], dtype=np.float32)
                        self.model_size = data.get("model", "medium")
                        self.language = data.get("language", "pl")
                        self.interval = float(data.get("interval", 6))

                        await self.send({"type": "status", "text": f"Loading model '{self.model_size}'..."})
                        self.model, self.infer_lock = await get_model(self.model_size)
                        self.running = True
                        await self.send({"type": "status", "text": "Ready. Recording..."})

                    elif action == "stop":
                        self.running = False
                        if len(self.audio_buffer) >= self.SAMPLE_RATE * 0.5:
                            await self._transcribe_buffer()
                        else:
                            self.audio_buffer = np.array([], dtype=np.float32)
                        await self.send({"type": "status", "text": "Stopped."})

                elif isinstance(message, bytes) and self.running:
                    chunk = np.frombuffer(message, dtype=np.float32)
                    self.audio_buffer = np.append(self.audio_buffer, chunk)

                    if len(self.audio_buffer) >= self.SAMPLE_RATE * self.interval:
                        await self._transcribe_buffer()
        finally:
            self.running = False


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
