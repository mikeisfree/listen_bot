import React, { useState, useEffect, useRef, useCallback } from "react";
import "./index.css";

interface WsMessage {
  type: "status" | "transcript" | "error" | "keyword";
  text?: string;
  message?: string;
  keyword?: string;
}

interface SavedSession {
  id: string;
  date: string;
  language: string;
  source?: string;
  segments: string[];
}

const SOURCE_LABELS: Record<string, string> = {
  system: "System",
  mic: "Mic",
};

type ViewMode = "live" | "history" | "detail";

const STORAGE_KEY = "listenbot_transcripts";
const MAX_SESSIONS = 50;

const LANGUAGE_OPTIONS: { value: string; label: string }[] = [
  { value: "pl", label: "PL — Polski" },
  { value: "en", label: "EN — English" },
  { value: "fr", label: "FR — Français" },
  { value: "es", label: "ES — Español" },
  { value: "it", label: "IT — Italiano" },
];

function getWsUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return (import.meta.env.VITE_WS_URL as string | undefined) ?? `${proto}//${window.location.host}/ws`;
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

function loadSessions(): SavedSession[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as SavedSession[]) : [];
  } catch {
    return [];
  }
}

function persistSessions(sessions: SavedSession[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions.slice(0, MAX_SESSIONS)));
  } catch {
    // localStorage full — silently ignore
  }
}

const App: React.FC = () => {
  // Settings
  const [model, setModel] = useState("small");
  const [source, setSource] = useState("system");
  const [language, setLanguage] = useState("pl");
  const [interval, setIntervalVal] = useState("6.0");

  // Connection & runtime state
  const [isRunning, setIsRunning] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [status, setStatus] = useState("Disconnected");
  const [keywordDetected, setKeywordDetected] = useState(false);
  const [micError, setMicError] = useState<string | null>(null);

  // Live transcript
  const [transcripts, setTranscripts] = useState<string[]>([]);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const lastTranscript = useRef<string | null>(null);
  const transcriptsRef = useRef<string[]>([]);

  // Session tracking (for saving)
  const activeSession = useRef<{ id: string; date: string; language: string; source: string } | null>(null);

  // WebSocket
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Audio capture — browser microphone → raw PCM → WebSocket
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Navigation
  const [viewMode, setViewMode] = useState<ViewMode>("live");
  const [savedSessions, setSavedSessions] = useState<SavedSession[]>(() => loadSessions());
  const [selectedSession, setSelectedSession] = useState<SavedSession | null>(null);

  // Keep ref in sync with transcript state (safe for use in effects/handlers)
  transcriptsRef.current = transcripts;

  useEffect(() => {
    if (transcriptRef.current) {
      transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight;
    }
  }, [transcripts]);

  const saveCurrentSession = useCallback(() => {
    const meta = activeSession.current;
    const segs = transcriptsRef.current;
    if (!meta || segs.length === 0) return;
    const session: SavedSession = { ...meta, segments: [...segs] };
    setSavedSessions((prev) => {
      const updated = [session, ...prev.filter((s) => s.id !== session.id)];
      persistSessions(updated);
      return updated;
    });
    activeSession.current = null;
  }, []);

  // Stop browser audio stream without touching WebSocket
  const stopAudio = useCallback(() => {
    processorRef.current?.disconnect();
    processorRef.current = null;
    void audioContextRef.current?.close();
    audioContextRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, []);

  // Auto-save when transcription stops
  useEffect(() => {
    if (!isRunning) saveCurrentSession();
  }, [isRunning, saveCurrentSession]);

  // Save on page close
  useEffect(() => {
    const handleUnload = () => saveCurrentSession();
    window.addEventListener("beforeunload", handleUnload);
    return () => window.removeEventListener("beforeunload", handleUnload);
  }, [saveCurrentSession]);

  const connect = useCallback(() => {
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }
    if (ws.current) {
      ws.current.onclose = null;
      ws.current.close();
    }
    ws.current = new WebSocket(getWsUrl());

    ws.current.onopen = () => {
      setIsConnected(true);
      setStatus("System Ready");
    };

    ws.current.onclose = () => {
      setIsConnected(false);
      setIsRunning(false);
      setStatus("Server Offline");
      stopAudio();
      reconnectTimer.current = setTimeout(connect, 3000);
    };

    ws.current.onmessage = (event) => {
      const data: WsMessage = JSON.parse(event.data as string);
      if (data.type === "status") {
        setStatus(data.text ?? "");
        if (data.text?.includes("Recording")) setIsRunning(true);
        if (data.text === "Stopped.") setIsRunning(false);
      } else if (data.type === "transcript") {
        const text = data.text ?? "";
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
  }, [stopAudio]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (ws.current) {
        ws.current.onclose = null;
        ws.current.close();
      }
    };
  }, [connect]);

  const startTranscription = async () => {
    setMicError(null);
    if (ws.current?.readyState !== WebSocket.OPEN) return;

    if (transcriptsRef.current.length > 0 && activeSession.current) {
      saveCurrentSession();
    }

    let stream: MediaStream;

    if (source === "system") {
      if (!navigator.mediaDevices?.getDisplayMedia) {
        setMicError("Screen capture not supported in this browser. Try Chrome or Edge.");
        return;
      }
      try {
        // Chrome requires video:true even when only audio is needed — stop video tracks immediately
        const display = await navigator.mediaDevices.getDisplayMedia({ audio: true, video: true });
        display.getVideoTracks().forEach((t) => t.stop());
        stream = display;
        if (stream.getAudioTracks().length === 0) {
          setMicError("No audio captured. Select a source and check 'Share audio' / 'Share tab audio'.");
          return;
        }
      } catch (err) {
        setMicError(`Capture cancelled: ${err instanceof Error ? err.message : String(err)}`);
        return;
      }
    } else {
      if (!navigator.mediaDevices?.getUserMedia) {
        setMicError("Microphone requires HTTPS.");
        return;
      }
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      } catch (err) {
        setMicError(`Mic blocked: ${err instanceof Error ? err.message : String(err)}`);
        return;
      }
    }

    streamRef.current = stream;

    // Request 16 kHz to match Whisper's native sample rate — no backend resampling needed
    const audioCtx = new AudioContext({ sampleRate: 16000 });
    audioContextRef.current = audioCtx;

    const audioSource = audioCtx.createMediaStreamSource(stream);
    // 4096 samples @ 16 kHz ≈ 256 ms per chunk
    const processor = audioCtx.createScriptProcessor(4096, 1, 1);
    processorRef.current = processor;

    processor.onaudioprocess = (e) => {
      if (ws.current?.readyState === WebSocket.OPEN) {
        // .slice() copies the Float32Array so we don't send a stale shared buffer
        const pcm = e.inputBuffer.getChannelData(0).slice();
        ws.current.send(pcm.buffer);
      }
    };

    audioSource.connect(processor);
    processor.connect(audioCtx.destination);

    const now = new Date().toISOString();
    activeSession.current = {
      id: now.replace(/[:.]/g, "-").slice(0, 19),
      date: now,
      language,
      source,
    };
    lastTranscript.current = null;
    setTranscripts([]);
    setViewMode("live");

    ws.current.send(
      JSON.stringify({ action: "start", model, language, interval: parseFloat(interval) })
    );
    setStatus("Starting...");
  };

  const stopTranscription = () => {
    stopAudio();
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ action: "stop" }));
    }
    setIsRunning(false);
  };

  const deleteSession = (id: string) => {
    setSavedSessions((prev) => {
      const updated = prev.filter((s) => s.id !== id);
      persistSessions(updated);
      return updated;
    });
    if (selectedSession?.id === id) {
      setSelectedSession(null);
      setViewMode("history");
    }
  };

  const downloadSession = (session: SavedSession) => {
    const langLabel = LANGUAGE_OPTIONS.find((o) => o.value === session.language)?.label ?? session.language;
    const langName = langLabel.split("—")[1]?.trim() ?? session.language;
    const srcLabel = SOURCE_LABELS[session.source ?? "system"] ?? session.source ?? "System";
    const content = [
      `# Transcript — ${formatDate(session.date)}`,
      `Language: ${langName} | Source: ${srcLabel}`,
      "",
      "---",
      "",
      ...session.segments,
    ].join("\n");
    const blob = new Blob([content], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `transcript_${session.id}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const contentTitle =
    viewMode === "detail"
      ? formatDate(selectedSession?.date ?? "")
      : viewMode === "history"
      ? "History"
      : "Notes";

  return (
    <div className="layout">
      <aside className="sidebar">
        <header className="sidebar-header">
          <h1 className="app-title">ListenBot</h1>
          <div className="status-indicator">
            <span className={`status-dot ${isConnected ? "connected" : "disconnected"}`} />
            <span className="status-text">{isConnected ? "Server Active" : "Offline"}</span>
          </div>
        </header>

        <form className="settings-form" onSubmit={(e) => e.preventDefault()}>
          <div className="field">
            <label htmlFor="source">Audio Source</label>
            <select id="source" value={source} onChange={(e) => setSource(e.target.value)} disabled={isRunning}>
              <option value="system">System Audio (tab / screen)</option>
              <option value="mic">Microphone</option>
            </select>
            <div className="field-help">System: picks browser tab or screen with audio.</div>
          </div>

          <div className="field">
            <label htmlFor="model">Whisper Model</label>
            <select id="model" value={model} onChange={(e) => setModel(e.target.value)} disabled={isRunning}>
              <option value="tiny">Tiny (Fastest)</option>
              <option value="base">Base (Balanced)</option>
              <option value="small">Small (Good Accuracy)</option>
              <option value="medium">Medium (Professional)</option>
            </select>
            <div className="field-help">Determines transcription speed vs accuracy.</div>
          </div>

          <div className="field">
            <label htmlFor="language">Language</label>
            <select id="language" value={language} onChange={(e) => setLanguage(e.target.value)} disabled={isRunning}>
              {LANGUAGE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <label htmlFor="interval">Transcription Delay (s)</label>
            <input
              id="interval"
              type="number"
              min="1"
              step="0.5"
              value={interval}
              onChange={(e) => setIntervalVal(e.target.value)}
              disabled={isRunning}
            />
            <div className="field-help">Time per chunk. Higher = longer lines.</div>
          </div>
        </form>

        <div className="sidebar-footer">
          {isRunning ? (
            <button className="btn btn-danger" onClick={stopTranscription}>
              Stop Session
            </button>
          ) : (
            <button className="btn btn-primary" onClick={startTranscription} disabled={!isConnected}>
              Start Recording
            </button>
          )}
          {micError && <p className="mic-error">{micError}</p>}
          {micError && <p className="error-notice">{micError}</p>}
          <button
            className={`btn btn-secondary${viewMode !== "live" ? " active" : ""}`}
            onClick={() => {
              if (viewMode !== "live") {
                setViewMode("live");
                setSelectedSession(null);
              } else {
                setViewMode("history");
              }
            }}
          >
            {viewMode !== "live"
              ? "Back to Live"
              : `My Transcripts${savedSessions.length > 0 ? ` (${savedSessions.length})` : ""}`}
          </button>
        </div>
      </aside>

      <main className="main-content">
        <header className="content-header">
          <div>
            <h2>{contentTitle}</h2>
            {viewMode === "detail" && selectedSession && (
              <div className="session-meta">
                {selectedSession.language.toUpperCase()} · {selectedSession.segments.length} segments
              </div>
            )}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            {viewMode === "live" && (
              <>
                <div className="status-badge" title={status}>
                  <span
                    className={`status-light ${
                      isRunning ? "pulse-recording" : isConnected ? "ready" : "offline"
                    }`}
                  />
                  <span className="status-label">
                    {isRunning ? "Recording" : isConnected ? "Ready" : "Offline"}
                  </span>
                </div>
                <div className="keyword-indicator" title="grab command indicator">
                  <span className={`keyword-dot ${keywordDetected ? "keyword-active" : ""}`} />
                  <span className="keyword-label">Grab</span>
                </div>
              </>
            )}
            {viewMode === "detail" && selectedSession && (
              <button className="btn-text" onClick={() => downloadSession(selectedSession)}>
                Download .md
              </button>
            )}
          </div>
        </header>

        {viewMode === "live" && (
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
        )}

        {viewMode === "history" && (
          <div className="history-list">
            {savedSessions.length === 0 ? (
              <div className="empty-state" style={{ paddingTop: "40px" }}>
                No saved transcripts yet.
              </div>
            ) : (
              savedSessions.map((session) => (
                <div key={session.id} className="history-item">
                  <div className="history-item-meta">
                    <span className="history-date">{formatDate(session.date)}</span>
                    <div className="history-tags">
                      <span className="tag">{session.language.toUpperCase()}</span>
                      <span className="tag">{SOURCE_LABELS[session.source ?? "system"] ?? session.source}</span>
                      <span className="tag">{session.segments.length} seg</span>
                    </div>
                  </div>
                  {session.segments.length > 0 && (
                    <div className="history-preview">
                      {session.segments.join(" ").slice(0, 140)}
                      {session.segments.join(" ").length > 140 ? "…" : ""}
                    </div>
                  )}
                  <div className="history-actions">
                    <button
                      className="btn-text"
                      onClick={() => {
                        setSelectedSession(session);
                        setViewMode("detail");
                      }}
                    >
                      View
                    </button>
                    <button className="btn-text" onClick={() => downloadSession(session)}>
                      Download
                    </button>
                    <button
                      className="btn-text btn-text-danger"
                      onClick={() => deleteSession(session.id)}
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {viewMode === "detail" && selectedSession && (
          <div className="transcript-container">
            <div className="transcript-list">
              {selectedSession.segments.map((seg, i) => (
                <p key={i} className="transcript-line">
                  {seg}
                </p>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
