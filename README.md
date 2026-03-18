# Listen BOT 🎙️

A real-time speech-to-text transcription tool optimized for Linux (PulseAudio/PipeWire) using **Faster-Whisper**. It captures system audio, transcribes it in real-time, saves it to Markdown files, and broadcasts the result via WebSockets to a web interface.

## 🚀 Features

- **Real-time Transcription**: Powered by OpenAI's Whisper (via `faster-whisper`).
- **System Audio Capture**: Automatically detects PulseAudio/PipeWire monitor devices on Linux.
- **GPU Acceleration**: Supports NVIDIA CUDA for high-performance transcription.
- **Auto-save**: Transcripts are automatically saved to the `Transcripts/` directory as `.md` files.
- **Modern Web Interface**: Built with React, TypeScript, and Vite for live text display.
- **CLI Support**: Start transcriptions directly from the terminal if needed.

## 🛠️ Setup

### Backend (Python)

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Check audio devices** (Optional):
   ```bash
   python list_devices.py
   ```

### Frontend (React/Vite)

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

## 🏃 Running the Application

1. **Start the Backend**:
   ```bash
   python backend.py
   ```
   *The backend will listen on `ws://localhost:8765`.*

2. **Start the Frontend**:
   ```bash
   cd frontend
   npm run dev
   ```
   *Open the provided URL (usually `http://localhost:5173`) in your browser.*

3. **Start Transcribing**:
   Enter a filename and click **Start** in the web interface.

## 📂 Project Structure

- `backend.py`: Main transcription engine and WebSocket server.
- `frontend/`: React web application.
- `Transcripts/`: Auto-generated transcription files.
- `list_devices.py`: Utility to list available audio input devices.

## ⚡ Tech Stack

- **Backend**: Python, Faster-Whisper, SoundDevice, WebSockets, NumPy.
- **Frontend**: React, TypeScript, Vite, CSS (Tailwind compatible).
- **Optimization**: NVIDIA CUDA (cuBLAS, cuDNN) for GPU support.

---

*Made with love for real-time note-taking.*
