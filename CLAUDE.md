# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Backend (Python)
- Run backend: `python backend.py`
- List audio devices: `python list_devices.py`
- Install dependencies: `pip install -r requirements.txt`

### Frontend (React/Vite)
- Install dependencies: `npm install` (in `frontend/`)
- Run dev server: `npm run dev` (in `frontend/`)
- Build frontend: `npm run build` (in `frontend/`)
- Lint frontend: `npm run lint` (in `frontend/`)

## Architecture

### High-Level Structure
The project is a real-time speech-to-text transcription tool consisting of a Python backend and a React frontend.

- **Backend (`backend.py`)**: 
  - Uses `faster-whisper` for transcription.
  - Manages audio streams via `sounddevice` (optimized for PulseAudio/PipeWire on Linux).
  - Implements a WebSocket server (`ws://localhost:8765`) to broadcast transcripts to clients.
  - Supports GPU acceleration via NVIDIA CUDA (paths are manually configured in `setup_cuda_paths`).
  - Saves transcripts to the `Transcripts/` directory as `.md` files.
- **Frontend (`frontend/`)**:
  - React/TypeScript application built with Vite.
  - Connects to the backend via WebSockets to display live transcription text.
  - Provides a UI to start/stop transcription, select audio sources (System/Mic/Both), and configure model sizes.
