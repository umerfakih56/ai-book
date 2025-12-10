import React, { useEffect, useRef, useState } from "react";
import useTextSelection from "./useTextSelection";

const BACKEND_URL =
  (typeof process !== "undefined" && process.env.DOCS_RAG_BACKEND_URL) ||
  "http://localhost:8000";

type Message = {
  role: "user" | "assistant";
  content: string;
};

function classNames(...classes: Array<string | boolean | null | undefined>) {
  return classes.filter(Boolean).join(" ");
}

const RAGChatbot: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [streamError, setStreamError] = useState<string | null>(null);
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
    setStreamError(null);

    const abortController = new AbortController();

    try {
      const response = await fetch(`${BACKEND_URL}/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question,
          selected_text: selectedText || null,
          top_k: 5,
        }),
        signal: abortController.signal,
      });

      if (!response.ok || !response.body) {
        throw new Error(`HTTP error ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const text = decoder.decode(value, { stream: true });
        const lines = text.split("\n\n");
        for (const line of lines) {
          if (!line.trim()) continue;
          if (line.startsWith("data:")) {
            const data = line.replace(/^data:\s*/, "");
            if (data === "[DONE]") {
              setIsLoading(false);
              clearSelection();
              return;
            }
            setMessages((prev) => {
              const updated = [...prev];
              const current = updated[assistantIndex];
              updated[assistantIndex] = {
                ...current,
                content: (current?.content || "") + data,
              };
              return updated;
            });
          }
        }
      }
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("Stream error", err);
      setStreamError("Error streaming response");
    } finally {
      setIsLoading(false);
    }
  };

  const handleClose = () => {
    setIsOpen(false);
    setStreamError(null);
  };

  return (
    <>
      <button
        type="button"
        onClick={() => setIsOpen(true)}
        className="fixed bottom-4 right-4 z-40 rounded-full bg-blue-600 text-white px-4 py-2 shadow-lg hover:bg-blue-700 focus:outline-none"
      >
        Ask AI
      </button>

      <div
        className={classNames(
          "fixed bottom-20 right-4 z-40 w-80 max-w-full bg-white border border-gray-200 rounded-lg shadow-xl flex flex-col",
          isOpen ? "opacity-100 pointer-events-auto" : "opacity-0 pointer-events-none"
        )}
      >
        <div className="flex items-center justify-between px-3 py-2 border-b border-gray-200 bg-gray-50">
          <div>
            <div className="font-semibold text-sm">Physical AI Assistant</div>
            {selectedText && (
              <div className="text-xs text-blue-600 mt-1">
                Using selected text as context
              </div>
            )}
          </div>
          <button
            type="button"
            onClick={handleClose}
            className="text-gray-500 hover:text-gray-700 text-sm"
          >
            ✕
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-3 py-2 text-sm">
          {messages.length === 0 && (
            <p className="text-gray-500 text-xs">
              Ask a question about the Physical AI &amp; Humanoid Robotics textbook,
              or select text in the chapter and then ask a question about it.
            </p>
          )}
          {messages.map((m, idx) => (
            <div
              key={idx}
              className={classNames(
                "my-1 whitespace-pre-wrap",
                m.role === "user" ? "text-right" : "text-left"
              )}
            >
              <span
                className={classNames(
                  "inline-block px-2 py-1 rounded",
                  m.role === "user"
                    ? "bg-blue-600 text-white"
                    : "bg-gray-100 text-gray-900"
                )}
              >
                {m.content}
              </span>
            </div>
          ))}
          {isLoading && (
            <div className="text-xs text-gray-500 mt-1">Thinking85</div>
          )}
          {streamError && (
            <div className="text-xs text-red-600 mt-1">{streamError}</div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="border-t border-gray-200 px-2 py-2">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
            className="flex items-center space-x-1"
          >
            <textarea
              rows={1}
              className="flex-1 border border-gray-300 rounded px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 resize-none"
              placeholder="Ask a question..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="bg-blue-600 text-white rounded px-3 py-1 text-sm disabled:opacity-50"
            >
              Send
            </button>
          </form>
        </div>
      </div>
    </>
  );
};

export default RAGChatbot;