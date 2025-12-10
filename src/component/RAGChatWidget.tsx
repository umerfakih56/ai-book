import React, { useEffect, useRef, useState } from "react";
import useTextSelection from "./useTextSelection";
import { TbMessageChatbot } from "react-icons/tb";
// Gemini backend URL (see rag-backend/simple_gemini_chat.py)
// Uses env var in production, falls back to localhost for local dev.
const BACKEND_URL =
  (typeof process !== "undefined" &&
    process.env.DOCS_RAG_BACKEND_URL) ||
  "http://127.0.0.1:9000";

type Message = {
  role: "user" | "assistant";
  content: string;
};

const containerStyle: React.CSSProperties = {
  position: "fixed",
  bottom: "80px",
  right: "16px",
  width: "320px",
  maxWidth: "100%",
  height: "360px", // fixed height for the chat panel
  backgroundColor: "#111827", // dark background
  border: "1px solid #374151",
  borderRadius: "12px",
  boxShadow: "0 10px 25px rgba(0,0,0,0.4)",
  display: "flex",
  flexDirection: "column",
  zIndex: 9999,
};

const headerStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  padding: "8px 12px",
  borderBottom: "1px solid #374151",
  backgroundColor: "#111827",
};

const messagesStyle: React.CSSProperties = {
  flex: 1,
  overflowY: "auto", // vertical scrollbar when content grows
  padding: "8px 12px",
  fontSize: "0.875rem",
  color: "#e5e7eb",
};

const inputBarStyle: React.CSSProperties = {
  borderTop: "1px solid #374151",
  padding: "8px 8px",
};

const floatingButtonStyle: React.CSSProperties = {
  position: "fixed",
  bottom: "16px",
  right: "16px",
  width: "56px",
  height: "56px",
  borderRadius: "9999px",
  backgroundColor: "#2563eb",
  color: "#ffffff",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  boxShadow: "0 10px 25px rgba(0,0,0,0.25)",
  border: "none",
  cursor: "pointer",
  zIndex: 9999,
  fontSize: "24px",
  lineHeight: 1,
};

const RAGChatWidget: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const { selectedText, hasSelection, clearSelection } = useTextSelection();

  useEffect(() => {
    if (hasSelection && !isOpen) {
      setIsOpen(true);
    }
  }, [hasSelection, isOpen]);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isLoading]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const question = input.trim();
    const assistantIndex = messages.length + 1;

    setMessages((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "" },
    ]);

    setInput("");
    setIsLoading(true);
    setErrorMessage(null);

    try {
      const response = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: question }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }

      const data = (await response.json()) as { reply?: string };
      const reply = data.reply ?? "";

      setMessages((prev) => {
        const updated = [...prev];
        const current = updated[assistantIndex];
        updated[assistantIndex] = {
          ...current,
          content: reply,
        };
        return updated;
      });

      clearSelection();
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("Chat error", err);
      setErrorMessage("Error contacting Gemini backend");
    } finally {
      setIsLoading(false);
    }
  };

  const handleClose = () => {
    setIsOpen(false);
    setErrorMessage(null);
  };

  return (
    <>
      {/* Sticky bottom-right circular button */}
      <button
        type="button"
        style={floatingButtonStyle}
        onClick={() => setIsOpen(true)}
        aria-label="Open Physical AI chatbot"
      >
  <TbMessageChatbot />
      </button>

      {/* Chat container */}
      {isOpen && (
        <div style={containerStyle}>
          <div style={headerStyle}>
            <div>
              <div
                style={{
                  fontWeight: 600,
                  fontSize: "0.875rem",
                  color: "#f9fafb",
                }}
              >
                AI Assistant
              </div>
              {selectedText && (
                <div
                  style={{
                    fontSize: "0.75rem",
                    color: "#9ca3af",
                    marginTop: 4,
                  }}
                >
                  Using selected text as context
                </div>
              )}
            </div>
            <button
              type="button"
              onClick={handleClose}
              style={{
                border: "none",
                background: "transparent",
                cursor: "pointer",
                color: "#6b7280",
                fontSize: "16px",
              }}
              aria-label="Close chatbot"
            >
              ×
            </button>
          </div>

          <div style={messagesStyle}>
            {messages.length === 0 && (
              <p
                style={{ color: "#9ca3af", fontSize: "0.75rem" }}
              >
                Ask a question about the Physical AI & Humanoid Robotics textbook,
                or select text in the chapter and then ask a question about it.
              </p>
            )}
            {messages.map((m, idx) => (
              <div
                key={idx}
                style={{
                  textAlign: m.role === "user" ? "right" : "left",
                  margin: "4px 0",
                  whiteSpace: "pre-wrap",
                }}
              >
                <span
                  style={{
                    display: "inline-block",
                    padding: "4px 8px",
                    borderRadius: 8,
                    backgroundColor:
                      m.role === "user" ? "#2563eb" : "#1f2937",
                    color: m.role === "user" ? "#ffffff" : "#e5e7eb",
                  }}
                >
                  {m.content}
                </span>
              </div>
            ))}
            {isLoading && (
              <div
                style={{
                  fontSize: "0.75rem",
                  color: "#9ca3af",
                  marginTop: 4,
                }}
              >
                Thinking...
              </div>
            )}
            {errorMessage && (
              <div
                style={{
                  fontSize: "0.75rem",
                  color: "#f87171",
                  marginTop: 4,
                }}
              >
                {errorMessage}
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div style={inputBarStyle}>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSend();
              }}
              style={{ display: "flex", gap: 4, alignItems: "center" }}
            >
              <textarea
                rows={1}
                style={{
                  flex: 1,
                  borderRadius: 6,
                  border: "1px solid #4b5563",
                  backgroundColor: "#111827",
                  color: "#e5e7eb",
                  padding: "4px 6px",
                  fontSize: "0.875rem",
                  resize: "none",
                }}
                placeholder="Ask a question..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
              />
              <button
                type="submit"
                disabled={!input.trim() || isLoading}
                style={{
                  borderRadius: 6,
                  border: "none",
                  padding: "4px 10px",
                  fontSize: "0.875rem",
                  backgroundColor: "#2563eb",
                  color: "#ffffff",
                  cursor:
                    !input.trim() || isLoading ? "not-allowed" : "pointer",
                  opacity: !input.trim() || isLoading ? 0.6 : 1,
                }}
              >
                Send
              </button>
            </form>
          </div>
        </div>
      )}
    </>
  );
};

export default RAGChatWidget;