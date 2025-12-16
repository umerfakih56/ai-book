import React, { useState, useEffect } from 'react';

// Cache utility functions
const getCachedTranslation = (text) => {
  try {
    const cache = localStorage.getItem('translationCache');
    return cache ? JSON.parse(cache)[text] : null;
  } catch (e) {
    console.error('Error reading from cache:', e);
    return null;
  }
};

const cacheTranslation = (text, translation) => {
  try {
    const cache = JSON.parse(localStorage.getItem('translationCache') || '{}');
    cache[text] = translation;
    localStorage.setItem('translationCache', JSON.stringify(cache));
  } catch (e) {
    console.error('Error writing to cache:', e);
  }
};

const TranslateButton = ({ text }) => {
  const [loading, setLoading] = useState(false);
  const [showTranslation, setShowTranslation] = useState(false);
  const [translation, setTranslation] = useState('');
  const [error, setError] = useState('');

  // Check for cached translation when component mounts
  useEffect(() => {
    if (text) {
      const cached = getCachedTranslation(text);
      if (cached) {
        setTranslation(cached);
      }
    }
  }, [text]);

  const handleTranslate = async () => {
    if (!text) {
      setError('No text to translate');
      return;
    }

    // Check cache first
    const cached = getCachedTranslation(text);
    if (cached) {
      console.log('Using cached translation');
      setTranslation(cached);
      setShowTranslation(true);
      return;
    }

    setLoading(true);
    setError('');
    console.log('Sending translation request for text:', text.substring(0, 50) + '...');

    try {
      const response = await fetch('http://localhost:3000/api/gemini-translate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      console.log('Response status:', response.status);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Translation response:', data);
      
      if (!data.translation) {
        throw new Error('No translation in response');
      }

      // Cache the translation
      cacheTranslation(text, data.translation);
      
      setTranslation(data.translation);
      setShowTranslation(true);
    } catch (err) {
      console.error('Translation error:', err);
      setError(`Translation failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ 
      margin: '20px 0',
      position: 'relative',
      zIndex: 1000
    }}>
      <button
        onClick={() => {
          if (showTranslation) {
            setShowTranslation(false);
          } else {
            handleTranslate();
          }
        }}
        disabled={loading}
        style={{
          backgroundColor: '#2e8555',
          color: 'white',
          border: 'none',
          padding: '10px 20px',
          borderRadius: '5px',
          cursor: loading ? 'not-allowed' : 'pointer',
          fontSize: '16px',
          fontWeight: 'bold',
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}
      >
        {loading ? (
          <>
            <span className="spinner"></span>
            ترجمہ ہو رہا ہے...
          </>
        ) : showTranslation ? (
          'ترجمہ چھپائیں'
        ) : (
          'اردو میں ترجمہ کریں'
        )}
      </button>

      {error && (
        <div style={{ 
          color: '#e74c3c', 
          marginTop: '10px',
          padding: '10px',
          backgroundColor: '#fdecea',
          borderRadius: '4px',
          border: '1px solid #f5c6cb'
        }}>
          {error}
        </div>
      )}

      {showTranslation && translation && (
        <div style={{ 
          marginTop: '15px',
          padding: '20px',
          backgroundColor: '#f0fdf4',
          border: '1px solid #86efac',
          borderRadius: '8px',
          color: '#166534',
          direction: 'rtl',
          textAlign: 'right',
          fontFamily: '"Noto Naskh Arabic", serif',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          position: 'relative'
        }}>
          <button
            onClick={() => setShowTranslation(false)}
            style={{
              position: 'absolute',
              top: '10px',
              left: '10px',
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              fontSize: '14px',
              color: '#666'
            }}
            title="Close"
          >
            ×
          </button>
          <div style={{ whiteSpace: 'pre-wrap' }}>{translation}</div>
          <div style={{ marginTop: '15px', display: 'flex', gap: '10px' }}>
            <button
              onClick={() => {
                navigator.clipboard.writeText(translation);
                // You might want to add a "Copied!" feedback here
              }}
              style={{
                background: '#2e8555',
                color: 'white',
                border: 'none',
                padding: '5px 10px',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '14px'
              }}
            >
              کاپی کریں
            </button>
          </div>
        </div>
      )}

      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        .spinner {
          display: inline-block;
          width: 16px;
          height: 16px;
          border: 2px solid rgba(255,255,255,0.3);
          border-radius: 50%;
          border-top-color: #fff;
          animation: spin 1s ease-in-out infinite;
        }
      `}</style>
    </div>
  );
};

export default TranslateButton;