import React, { useState, useEffect, useRef } from 'react';
import './index.css';

interface Message {
  type: 'status' | 'transcript' | 'error';
  text?: string;
  message?: string;
  filename?: string;
}

const App: React.FC = () => {
  const [model, setModel] = useState('small');
  const [source, setSource] = useState('system');
  const [filename, setFilename] = useState('transcript_web');
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState('Disconnected');
  const [isConnected, setIsConnected] = useState(false);
  const [transcripts, setTranscripts] = useState<string[]>([]);
  const ws = useRef<WebSocket | null>(null);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const lastTranscript = useRef<string | null>(null);

  useEffect(() => {
    connect();
    return () => ws.current?.close();
  }, []);

  useEffect(() => {
    if (transcriptRef.current) {
      transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight;
    }
  }, [transcripts]);

  const connect = () => {
    if (ws.current) ws.current.close();
    ws.current = new WebSocket('ws://localhost:8765');

    ws.current.onopen = () => {
      setIsConnected(true);
      setStatus('System Ready');
    };

    ws.current.onclose = () => {
      setIsConnected(false);
      setStatus('Server Offline');
      setTimeout(connect, 3000);
    };

    ws.current.onmessage = (event) => {
      const data: Message = JSON.parse(event.data);
      if (data.type === 'status') {
        setStatus(data.text || '');
        if (data.text?.includes('Recording')) setIsRunning(true);
        if (data.text === 'Stopped.') setIsRunning(false);
      } else if (data.type === 'transcript') {
        const text = data.text || '';
        // Prevent accidental duplicate from double-mount or same buffer processing
        if (text !== lastTranscript.current) {
          setTranscripts(prev => [...prev.slice(-100), text]);
          lastTranscript.current = text;
          setIsRunning(true);
        }
      } else if (data.type === 'error') {
        setStatus(`Error: ${data.message}`);
        setIsRunning(false);
      }
    };
  };

  const startTranscription = () => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({
        action: 'start',
        model: model,
        filename: filename,
        source: source
      }));
      setTranscripts([]);
      lastTranscript.current = null;
      setStatus('Starting Server...');
    }
  };

  const stopTranscription = () => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ action: 'stop' }));
      setIsRunning(false);
    }
  };

  return (
    <div className="glass">
      <div className="status-bar">
        <h1 style={{ fontSize: '1.5rem', fontWeight: 600, margin: 0 }}>ListenBot STT</h1>
        <div className={`badge ${isConnected ? 'badge-connected' : 'badge-disconnected'}`}>
          {isConnected ? 'SERVER ACTIVE' : 'RECONNECTING...'}
        </div>
      </div>

      <div className="controls-grid">
        <div>
          <label className="label">Whisper Model</label>
          <select 
            className="select-field" 
            value={model} 
            onChange={(e) => setModel(e.target.value)}
            disabled={isRunning}
          >
            <option value="tiny">Tiny (Fastest)</option>
            <option value="base">Base (Balanced)</option>
            <option value="small">Small (Good accuracy)</option>
            <option value="medium">Medium (Professional)</option>
          </select>
        </div>

        <div>
          <label className="label">Audio Source</label>
          <select 
            className="select-field" 
            value={source} 
            onChange={(e) => setSource(e.target.value)}
            disabled={isRunning}
          >
            <option value="system">System Audio (Monitor)</option>
            <option value="mic">Microphone (Input)</option>
          </select>
        </div>

        <div className="full-width">
          <label className="label">Markdown Filename</label>
          <input 
            className="input-field"
            type="text" 
            placeholder="e.g. daily_meeting"
            value={filename}
            onChange={(e) => setFilename(e.target.value)}
            disabled={isRunning}
          />
        </div>

        <div className="full-width">
          {isRunning ? (
            <button className="btn btn-stop full-width" onClick={stopTranscription}>
              <span className="recording-indicator"></span> Stop Session & Save
            </button>
          ) : (
            <button 
              className="btn btn-primary full-width" 
              onClick={startTranscription}
              disabled={!isConnected}
            >
              Start Recording
            </button>
          )}
        </div>
      </div>

      <div className="label">
        Status: <span style={{ color: isRunning ? 'var(--success)' : 'var(--text)' }}>{status}</span>
      </div>

      <div className="transcript-area" ref={transcriptRef}>
        {transcripts.length === 0 ? (
          <div className="text-dim" style={{ textAlign: 'center', marginTop: '10%' }}>
            {isRunning ? 'Analyzing audio...' : 'Press Start to begin transcription...'}
          </div>
        ) : (
          transcripts.map((line, i) => (
            <div key={i} className="transcript-line">
              {line}
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default App;
