import React, { useState, useEffect, useRef } from 'react';
import './ChatBot.css';
import RobotIcon from './RobotIcon';

const ChatBot = ({ apiUrl = 'http://localhost:8000' }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [sessionId, setSessionId] = useState('');
  const [error, setError] = useState('');
  const [selectionMode, setSelectionMode] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const [showSelectionTooltip, setShowSelectionTooltip] = useState(false);
  const [selectionPosition, setSelectionPosition] = useState({ x: 0, y: 0 });
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Initialize session ID from localStorage or create new one
  useEffect(() => {
    let storedSessionId = localStorage.getItem('chatbot_session_id');
    if (!storedSessionId) {
      storedSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      localStorage.setItem('chatbot_session_id', storedSessionId);
    }
    setSessionId(storedSessionId);

    // Load chat history
    loadChatHistory(storedSessionId);
  }, []);

  // Auto-scroll to latest message
  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  // Focus input when chat opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  // Handle text selection
  useEffect(() => {
    const handleSelection = () => {
      if (!selectionMode) return;
      
      const selection = window.getSelection();
      const text = selection.toString().trim();
      
      if (text.length > 10) {
        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        
        setSelectedText(text);
        setSelectionPosition({
          x: rect.left + rect.width / 2,
          y: rect.top - 10
        });
        setShowSelectionTooltip(true);
      } else {
        setShowSelectionTooltip(false);
        setSelectedText('');
      }
    };

    const handleSelectionChange = () => {
      handleSelection();
    };

    const handleDocumentClick = (e) => {
      // Hide tooltip if clicking outside
      if (!e.target.closest('.selection-tooltip') && !e.target.closest('.chat-window')) {
        setShowSelectionTooltip(false);
      }
    };

    if (selectionMode) {
      document.addEventListener('mouseup', handleSelectionChange);
      document.addEventListener('selectionchange', handleSelectionChange);
      document.addEventListener('click', handleDocumentClick);
    }

    return () => {
      document.removeEventListener('mouseup', handleSelectionChange);
      document.removeEventListener('selectionchange', handleSelectionChange);
      document.removeEventListener('click', handleDocumentClick);
    };
  }, [selectionMode]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const loadChatHistory = async (sid) => {
    try {
      const response = await fetch(`${apiUrl}/api/history/${sid}`);
      if (response.ok) {
        const history = await response.json();
        const formattedMessages = history.reverse().map(item => [
          { type: 'user', text: item.question, timestamp: item.created_at },
          { type: 'bot', text: item.answer, timestamp: item.created_at, sources: item.sources }
        ]).flat();
        setMessages(formattedMessages);
      }
    } catch (error) {
      console.error('Failed to load chat history:', error);
    }
  };

  const sendMessage = async (useSelectedText = false) => {
    if (!inputValue.trim() || isTyping) return;

    const userMessage = { type: 'user', text: inputValue, timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    const questionText = inputValue;
    setInputValue('');
    setIsTyping(true);
    setError('');

    try {
      const endpoint = useSelectedText ? '/api/chat/selected' : '/api/chat';
      const requestBody = useSelectedText ? {
        question: questionText,
        selected_text: selectedText,
        session_id: sessionId
      } : {
        question: questionText,
        session_id: sessionId
      };

      const response = await fetch(`${apiUrl}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();
      const botMessage = {
        type: 'bot',
        text: data.answer,
        timestamp: new Date(),
        sources: data.sources,
        mode: data.mode
      };
      setMessages(prev => [...prev, botMessage]);
      
      // Clear selection after sending
      if (useSelectedText) {
        clearSelection();
      }
    } catch (error) {
      setError('Failed to connect to AI assistant. Please try again.');
      console.error('Chat API error:', error);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setError('');
  };

  const clearSelection = () => {
    setSelectedText('');
    setShowSelectionTooltip(false);
    window.getSelection().removeAllRanges();
  };

  const toggleSelectionMode = () => {
    setSelectionMode(!selectionMode);
    if (selectionMode) {
      clearSelection();
    }
  };

  const handleSelectionTooltipClick = () => {
    const preview = selectedText.length > 100 
      ? selectedText.substring(0, 100) + '...' 
      : selectedText;
    setInputValue(`Based on the selected text: "${preview}"`);
    setShowSelectionTooltip(false);
    setIsOpen(true);
    setTimeout(() => {
      inputRef.current?.focus();
    }, 300);
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  const minimizeChat = () => {
    setIsOpen(false);
  };

  return (
    <div className="chatbot-container">
      {/* Floating Chat Button */}
      <button
        className={`chat-toggle-btn ${isOpen ? 'active' : ''}`}
        onClick={toggleChat}
        aria-label="Toggle chat"
      >
        <div className="robot-icon">
          <RobotIcon size={24} />
        </div>
        <span className="chat-badge">AI Assistant</span>
                <div className="tooltip">Ask me anything about Physical AI!</div>
      </button>

      {/* Chat Window */}
      <div className={`chat-window ${isOpen ? 'open' : ''}`}>
        {/* Header */}
        <div className="chat-header">
          <div className="chat-title">
            <div className="header-robot-icon">
              <RobotIcon size={20} />
            </div>
            Humanoid Robotics AI Tutor
          </div>
          <div className="chat-controls">
            <button 
              className={`control-btn selection-mode-btn ${selectionMode ? 'active' : ''}`} 
              onClick={toggleSelectionMode} 
              aria-label="Toggle selection mode"
              title={selectionMode ? "Disable selection mode" : "Enable selection mode"}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M14.17 3L8.08 3L3.17 7.41V21H20.83V3H14.17ZM12 19C10.34 19 9 17.66 9 16C9 14.34 10.34 13 12 13C13.66 13 15 14.34 15 16C15 17.66 13.66 19 12 19ZM15 10H9V5H15V10Z" fill="currentColor"/>
              </svg>
            </button>
            <button className="control-btn minimize-btn" onClick={minimizeChat} aria-label="Minimize">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M19 13H5V11H19V13Z" fill="currentColor"/>
              </svg>
            </button>
            <button className="control-btn close-btn" onClick={toggleChat} aria-label="Close">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M19 6.41L17.59 5L12 10.59L6.41 5L5 6.41L10.59 12L5 17.59L6.41 19L12 13.41L17.59 19L19 17.59L13.41 12L19 6.41Z" fill="currentColor"/>
              </svg>
            </button>
          </div>
        </div>

        {/* Messages Area */}
        <div className="chat-messages">
          {/* Selection Mode Instructions */}
          {selectionMode && (
            <div className="selection-instructions">
              <div className="selection-icon">📝</div>
              <div className="selection-text">
                <strong>Selection Mode Active</strong>
                <p>Select text on the page, then ask questions about it</p>
              </div>
              {selectedText && (
                <button className="clear-selection-btn" onClick={clearSelection}>
                  Clear Selection
                </button>
              )}
            </div>
          )}
          
          {messages.length === 0 && !isTyping && (
            <div className="welcome-message">
              <div className="welcome-avatar">
                <RobotIcon size={48} />
              </div>
              <div className="welcome-text">
                <p>Hello! I'm your Humanoid Robotics AI tutor.</p>
                <p>Ask me anything about ROS 2, NVIDIA Isaac, VLA, or any Physical AI topics!</p>
              </div>
            </div>
          )}
          
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.type}`}>
              <div className="message-avatar">
                {message.type === 'bot' ? (
                  <RobotIcon size={24} />
                ) : (
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 12C14.21 12 16 10.21 16 8C16 5.79 14.21 4 12 4C9.79 4 8 5.79 8 8C8 10.21 9.79 12 12 12ZM12 14C9.33 14 4 15.34 4 18V20H20V18C20 15.34 14.67 14 12 14Z" fill="currentColor"/>
                  </svg>
                )}
              </div>
              <div className="message-content">
                <div className="message-text">{message.text}</div>
                {message.mode === 'selected_text' && (
                  <div className="selection-badge">
                    📄 Answer based on selected text only
                  </div>
                )}
                {message.sources && message.sources.length > 0 && (
                  <div className="message-sources">
                    <small>Sources:</small>
                    {message.sources.map((source, idx) => (
                      <div key={idx} className="source-item">
                        <small>📄 {source.title} ({source.file})</small>
                      </div>
                    ))}
                  </div>
                )}
                <div className="message-time">
                  {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            </div>
          ))}

          {/* Typing Indicator */}
          {isTyping && (
            <div className="message bot typing">
              <div className="message-avatar">
                <RobotIcon size={24} />
              </div>
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="error-message">
              <small>⚠️ {error}</small>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="chat-input-area">
          <div className="input-container">
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about ROS 2, NVIDIA Isaac, VLA..."
              className="chat-input"
              disabled={isTyping}
              maxLength={1000}
            />
            <button
              className="send-btn"
              onClick={() => sendMessage(selectedText.length > 0)}
              disabled={!inputValue.trim() || isTyping}
              aria-label="Send message"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M2.01 21L23 12L2.01 3L2 10L17 12L2 14L2.01 21Z" fill="currentColor"/>
              </svg>
            </button>
          </div>
          <div className="input-footer">
            <button className="clear-btn" onClick={clearChat} disabled={messages.length === 0}>
              Clear chat
            </button>
            <span className="char-counter">
              {inputValue.length}/1000
            </span>
          </div>
        </div>
      </div>

      {/* Selection Tooltip */}
      {showSelectionTooltip && (
        <div 
          className="selection-tooltip"
          style={{
            left: `${selectionPosition.x}px`,
            top: `${selectionPosition.y}px`,
            transform: 'translate(-50%, -100%)'
          }}
        >
          <div className="tooltip-content">
            <span>📝 Ask about this selection</span>
            <button 
              className="tooltip-action-btn"
              onClick={handleSelectionTooltipClick}
            >
              Ask Question
            </button>
          </div>
          <div className="tooltip-arrow"></div>
        </div>
      )}
    </div>
  );
};

export default ChatBot;
