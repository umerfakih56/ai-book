import React, { useState } from 'react';

export default function ChatPage() {
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  async function send() {
    if (!question.trim()) return;
    const q = question.trim();
    setMessages(m => [...m, { role: 'user', text: q }]);
    setQuestion('');
    setLoading(true);
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      });
      const data = await res.json();
      if (data.answer) {
        setMessages(m => [...m, { role: 'assistant', text: data.answer, sources: data.sources }]);
      } else if (data.error) {
        setMessages(m => [...m, { role: 'assistant', text: `Error: ${data.error}` }]);
      }
    } catch (err) {
      setMessages(m => [...m, { role: 'assistant', text: `Error: ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ padding: 20, maxWidth: 900, margin: '0 auto' }}>
      <h1>Course RAG Chat</h1>
      <div style={{ border: '1px solid #ddd', padding: 12, minHeight: 200, borderRadius: 6, background: '#fff' }}>
        {messages.map((m, i) => (
          <div key={i} style={{ marginBottom: 12 }}>
            <strong style={{ display: 'block' }}>{m.role === 'user' ? 'You' : 'Assistant'}</strong>
            <div style={{ whiteSpace: 'pre-wrap' }}>{m.text}</div>
            {m.sources && m.sources.length > 0 && (
              <div style={{ marginTop: 6, fontSize: 12, color: '#666' }}>Sources: {m.sources.join(', ')}</div>
            )}
          </div>
        ))}
      </div>

      <div style={{ marginTop: 12, display: 'flex', gap: 8 }}>
        <input
          value={question}
          onChange={e => setQuestion(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') send(); }}
          placeholder="Ask a question about the course..."
          style={{ flex: 1, padding: '8px 12px', fontSize: 16 }}
          disabled={loading}
        />
        <button onClick={send} disabled={loading} style={{ padding: '8px 16px' }}>{loading ? 'Thinking...' : 'Send'}</button>
      </div>
    </div>
  );
}
