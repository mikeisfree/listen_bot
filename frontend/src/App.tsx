import React, { useState, useEffect, useRef } from "react";
import "./index.css";

interface Message {
  type: "status" | "transcript" | "error";
  text?: string;
  message?: string;
  filename?: string;
}

const App: React.FC = () => {
  const [model, setModel] = useState("small");
  const [source, setSource] = useState("system");
  const [filename, setFilename] = useState("transcript_web");
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState("Disconnected");
  const [isConnected, setIsConnected] = useState(false);
  const [transcripts, setTranscripts] = useState<string[]>([]);
  const [keywordDetected, setKeywordDetected] = useState(false);
  const ws = useRef<WebSocket | null>(null);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const lastTranscript = useRef<string | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    connect();
    return () => {
      // Clear pending reconnect timer
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
        reconnectTimer.current = null;
      }
      // Prevent onclose from scheduling a reconnect during unmount
      if (ws.current) {
        ws.current.onclose = null;
        ws.current.close();
      }
    };
  }, []);

  useEffect(() => {
    if (transcriptRef.current) {
      transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight;
    }
  }, [transcripts]);

  const connect = () => {
    // Clear any pending reconnect timer
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }
    if (ws.current) {
      ws.current.onclose = null; // Prevent this close from scheduling another reconnect
      ws.current.close();
    }
    ws.current = new WebSocket("ws://localhost:8765");

    ws.current.onopen = () => {
      setIsConnected(true);
      setStatus("System Ready");
    };

    ws.current.onclose = () => {
      setIsConnected(false);
      setStatus("Server Offline");
      reconnectTimer.current = setTimeout(connect, 3000);
    };

    ws.current.onmessage = (event) => {
      const data: Message | any = JSON.parse(event.data);
      if (data.type === "status") {
        setStatus(data.text || "");
        if (data.text?.includes("Recording")) setIsRunning(true);
        if (data.text === "Stopped.") setIsRunning(false);
      } else if (data.type === "transcript") {
        const text = data.text || "";
        if (text !== lastTranscript.current) {
          setTranscripts((prev) => [...prev.slice(-100), text]);
          lastTranscript.current = text;
          setIsRunning(true);
        }
      } else if (data.type === "keyword") {
        if (data.keyword === "grab") {
          setKeywordDetected(true);
          setStatus("Key command: grab detected");
          setTimeout(() => setKeywordDetected(false), 1200);
        }
      } else if (data.type === "error") {
        setStatus(`Error: ${data.message}`);
        setIsRunning(false);
      }
    };
  };

  const startTranscription = () => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(
        JSON.stringify({
          action: "start",
          model: model,
          filename: filename,
          source: source,
        }),
      );
      setTranscripts([]);
      lastTranscript.current = null;
      setStatus("Starting Server...");
    }
  };

  const stopTranscription = () => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ action: "stop" }));
      setIsRunning(false);
    }
  };

  return (
    <div className="layout">
      <aside className="sidebar">
        <header className="sidebar-header">
          <h1 className="app-title">ListenBot</h1>
          <div className="status-indicator">
            <span
              className={`status-dot ${isConnected ? "connected" : "disconnected"}`}
            ></span>
            <span className="status-text">
              {isConnected ? "Server Active" : "Offline"}
            </span>
          </div>
        </header>

        <form className="settings-form" onSubmit={(e) => e.preventDefault()}>
          <div className="field">
            <label htmlFor="model">Whisper Model</label>
            <select
              id="model"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={isRunning}
            >
              <option value="tiny">Tiny (Fastest)</option>
              <option value="base">Base (Balanced)</option>
              <option value="small">Small (Good Accuracy)</option>
              <option value="medium">Medium (Professional)</option>
            </select>
            <div className="field-help">
              Determines transcription speed vs accuracy.
            </div>
          </div>

          <div className="field">
            <label htmlFor="source">Audio Source</label>
            <select
              id="source"
              value={source}
              onChange={(e) => setSource(e.target.value)}
              disabled={isRunning}
            >
              <option value="system">System Audio (Monitor)</option>
              <option value="mic">Microphone (Input)</option>
              <option value="both">
                Both: System transcription + voice command
              </option>
            </select>
            <div className="field-help">Source of the audio to transcribe.</div>
          </div>

          <div className="field">
            <label htmlFor="filename">Markdown Filename</label>
            <input
              id="filename"
              type="text"
              placeholder="e.g. daily_meeting"
              value={filename}
              onChange={(e) => setFilename(e.target.value)}
              disabled={isRunning}
            />
          </div>
        </form>

        <div className="sidebar-footer">
          {isRunning ? (
            <button className="btn btn-danger" onClick={stopTranscription}>
              Stop Session
            </button>
          ) : (
            <button
              className="btn btn-primary"
              onClick={startTranscription}
              disabled={!isConnected}
            >
              Start Recording
            </button>
          )}
        </div>
      </aside>

      <main className="main-content">
        <header className="content-header">
          <h2>Notes</h2>
          <div className="status-badge" title={status}>
            <span
              className={`status-light ${isRunning ? "pulse-recording" : isConnected ? "ready" : "offline"}`}
            ></span>
            <span className="status-label">
              {isRunning ? "Recording" : isConnected ? "Ready" : "Offline"}
            </span>
          </div>
          <div className="keyword-indicator" title="grab command indicator">
            <span
              className={`keyword-dot ${keywordDetected ? "keyword-active" : ""}`}
            ></span>
            <span className="keyword-label">Grab</span>
          </div>
        </header>

        <div className="transcript-container" ref={transcriptRef}>
          {transcripts.length === 0 ? (
            <div className="empty-state">
              {isRunning
                ? "Listening and analyzing..."
                : "Press Start Recording to begin transcription."}
            </div>
          ) : (
            <div className="transcript-list">
              {transcripts.map((line, i) => (
                <p key={i} className="transcript-line">
                  {line}
                </p>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default App;
