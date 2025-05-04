// ✅ src/App.jsx
import React, { useState, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import ReactMarkdown from 'react-markdown';

// 1) Define all your possible welcome texts
const WELCOME_TEXTS = [
  "Hi! I'm LuddyBuddy, always happy to chat! 😊",
  "Hello there! LuddyBuddy at your service. How can I help today?",
  "Hey! LuddyBuddy here—ready to answer your questions!",
  "Greetings! I'm LuddyBuddy. What would you like to know?",
  "Hey there! LuddyBuddy reporting for duty! 🤖✨",
  "Hiya! LuddyBuddy here - what can I do for you today? 🌟",
  "Hello, friend! LuddyBuddy at your fingertips. 🖐️😊",
  "Yo! LuddyBuddy in the house—ask me anything! 🏠💬",
  "Hi, I’m LuddyBuddy—your pocket‑sized helper! 🎒🤗",
  "👋 Hey! LuddyBuddy’s on call—how can I assist?",
  "What’s up? LuddyBuddy ready to lend a hand! 🙋‍♂️👍",
  "Hello, hello! LuddyBuddy here—let’s chat!",
  "Hey, I’m LuddyBuddy—your friendly neighborhood helper! 🌟👋",
  "Hi! LuddyBuddy’s all ears—got questions? 👂❓",
  "Good day! LuddyBuddy at your service—shoot away! 🎯😊"
];

// 2) Utility to grab one at random
function getRandomWelcome() {
  const i = Math.floor(Math.random() * WELCOME_TEXTS.length);
  return WELCOME_TEXTS[i];
}

export default function App() {
  const sessionIdRef = useRef(uuidv4());
  const [messages, setMessages] = useState([
    {
      sender: 'bot',
      text: getRandomWelcome()
    }
]);
  const [input, setInput] = useState('');
  const [typing, setTyping] = useState(false);
  const [isOpen, setIsOpen] = useState(false);

// 4) (Optional) if you want a new greeting *every time* the user re-opens:
const handleOpen = () => {
  if (messages.length === 0) {
    setMessages([{
      sender: 'bot',
      text: getRandomWelcome()
    }]);
  }
  setIsOpen(true);
};

  const handleSend = async () => {
    if (!input.trim()) return;
    const newMessages = [...messages, { sender: 'user', text: input }];
    setMessages(newMessages);
    setInput('');
    setTyping(true);
    try {
      const res = await fetch('http://127.0.0.1:8000/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: input,
          session_id: sessionIdRef.current
        })
      });
      const data = await res.json();
      setMessages([...newMessages, { sender: 'bot', text: data.response || 'No response' }]);
    } catch {
      setMessages([...newMessages, { sender: 'bot', text: '⚠️ Error reaching the server.' }]);
    } finally {
      setTyping(false);
    }
  };

  const handleMinimize = () => {
    setIsOpen(false);
  };
  
  const handleCloseAndClear = async () => {
    try {
      await fetch('http://127.0.0.1:8000/exit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionIdRef.current })
      });
      localStorage.removeItem('luddy-session-id'); // optional
      setMessages([]);
    } catch (err) {
      console.error('Error on exit:', err);
    } finally {
      setIsOpen(false);
    }
  };
  return (
    <>
      {/* Floating Button */}
      <button
        onClick={handleOpen}
        style={{
          position: 'fixed',
          bottom: 24,
          right: 24,
          width: 64,
          height: 64,
          borderRadius: '50%',
          backgroundColor: '#990200',
          color: '#FFF',
          border: 'none',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
          cursor: 'pointer',
          zIndex: 9998
        }}
      >
        <span style={{ fontSize: 24 }}>🤖</span>
      </button>

      {/* Chat Window */}
      {isOpen && (
        <div
          style={{
            position: 'fixed',
            bottom: 24,
            right: 24,
            width: 400,
            height: '85vh',
            backgroundColor: '#FFF',
            border: '2px solid #000',
            borderRadius: 16,
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            boxShadow: '0 10px 15px rgba(0,0,0,0.1)',
            zIndex: 9999
          }}
        >
          {/* Header */}
          <div
            style={{
              backgroundColor: '#990200',
              color: '#FFF',
              padding: '16px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              borderTopLeftRadius: 14,
              borderTopRightRadius: 14
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <div
                style={{
                  width: 32,
                  height: 32,
                  backgroundColor: '#FFF',
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
              >
                🤖
              </div>
              <span style={{ fontSize: 16, fontWeight: 600 }}>LuddyBuddy</span>
              
            </div>
            {/* <button
              onClick={() => setIsOpen(false)}
              style={{
                background: 'none',
                border: 'none',
                color: '#FFF',
                fontSize: 20,
                cursor: 'pointer'
              }}
            >
              ×
            </button> */}
            {/* <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <button onClick={handleMinimize} title="Minimize"
                style={{
                  background: 'none', border: 'none', color: '#FFF',
                  fontSize: 16, marginRight: 8, cursor: 'pointer'
                }}>🔻</button>
              <button onClick={handleCloseAndClear} title="Close & Clear"
                style={{
                  background: 'none', border: 'none', color: '#FFF',
                  fontSize: 20, cursor: 'pointer'
                }}>❌</button>
            </div> */}

<div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
  <button
    onClick={handleMinimize}
    title="Minimize"
    style={{
      backgroundColor: 'transparent',
      border: 'none',
      color: '#FFDADA',
      fontSize: '20px',
      cursor: 'pointer',
      borderRadius: '4px',
      padding: '4px 8px',
      transition: 'background 0.2s'
    }}
    onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#B30000'}
    onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
  >
    &minus;
  </button>

  <button
    onClick={handleCloseAndClear}
    title="Close & Clear"
    style={{
      backgroundColor: 'transparent',
      border: 'none',
      color: '#FFDADA',
      fontSize: '20px',
      cursor: 'pointer',
      borderRadius: '4px',
      padding: '4px 8px',
      transition: 'background 0.2s'
    }}
    onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#B30000'}
    onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
  >
    ✕
  </button>
</div>

          </div>

          



          {/* Messages */}
          <div
            style={{
              flex: 1,
              padding: '16px',
              overflowY: 'auto',
              backgroundColor: '#FFF'
            }}
          >
            {messages.map((msg, i) => {
              const isBot = msg.sender === 'bot';
              return (
                <div
                  key={i}
                  style={{
                    display: 'flex',
                    justifyContent: isBot ? 'flex-start' : 'flex-end',
                    marginBottom: 12,
                    gap: 8
                  }}
                >
                  {isBot && (
                    <div
                      style={{
                        width: 24,
                        height: 24,
                        backgroundColor: '#990200',
                        color: '#FFF',
                        borderRadius: '50%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: 10
                      }}
                    >
                      🤖
                    </div>
                  )}
                  {/* <div
                      style={{
                        backgroundColor: isBot ? '#F3F4F6' : '#990200',
                        color: isBot ? '#333' : '#FFF',
                        padding: '12px 12px 12px 16px',
                        borderRadius: 12,
                        maxWidth: '75%',
                        lineHeight: 1.5,
                        wordBreak: 'break-word',       // ✅ wraps long URLs
                        whiteSpace: 'pre-wrap',        // ✅ respects markdown line breaks
                        fontSize: '12px'               // ✅ smaller, clean font
                      }}
                    >
  <ReactMarkdown
    children={msg.text}
    components={{
      a: ({ node, ...props }) => (
        <a
          {...props}
          style={{
            color: '#990200',
            textDecoration: 'underline',
            wordBreak: 'break-word',
            overflowWrap: 'anywhere'
          }}
          target="_blank"
          rel="noopener noreferrer"
        />
      ),
      strong: ({ node, ...props }) => (
        <strong style={{ fontWeight: 'bold' }} {...props} />
      ),
      em: ({ node, ...props }) => (
        <em style={{ fontStyle: 'italic' }} {...props} />
      )
    }}
  />
</div> */}
<div
  style={{
    backgroundColor: isBot ? '#F3F4F6' : '#990200',
    color: isBot ? '#333' : '#FFF',
    padding: '12px 12px 12px 16px', // reduced right padding
    borderRadius: 12,
    maxWidth: '75%',
    lineHeight: 1.5,
    wordBreak: 'break-word',
    whiteSpace: 'pre-wrap',
    fontSize: '12px' // applies to outer container
  }}
>
  <ReactMarkdown
    components={{
      p: ({ node, ...props }) => (
        <p style={{ margin: 0, fontSize: '12px' }} {...props} />
      ),
      a: ({ node, ...props }) => (
        <a
          {...props}
          style={{
            color: '#990200',
            textDecoration: 'underline',
            wordBreak: 'break-word',
            overflowWrap: 'anywhere',
            fontSize: '12px'
          }}
          target="_blank"
          rel="noopener noreferrer"
        />
      ),
      strong: ({ node, ...props }) => (
        <strong style={{ fontWeight: 'bold', fontSize: '12px' }} {...props} />
      ),
      em: ({ node, ...props }) => (
        <em style={{ fontStyle: 'italic', fontSize: '12px' }} {...props} />
      ),
      ul: ({ node, ...props }) => (
        <ul
          style={{
            listStyleType: 'disc',
            paddingLeft: '20px',
            margin: '8px 0',
            fontSize: '12px',
            color: '#333'
          }}
          {...props}
        />
      ),
      ol: ({ node, ...props }) => (
        <ol
          style={{
            listStyleType: 'decimal',
            paddingLeft: '20px',
            margin: '8px 0',
            fontSize: '12px',
            color: '#333'
          }}
          {...props}
        />
      ),
      li: ({ node, ...props }) => (
        <li
          style={{
            marginBottom: '4px',
            lineHeight: '1.5',
          }}
          {...props}
        />
      ),
    }}
  >
    {msg.text}
  </ReactMarkdown>
</div>
                  {!isBot && (
                    <div style={{ width: 24, marginLeft: 8 }} />
                  )}
                </div>
              );
            })}

            {typing && (
              <div style={{ display: 'flex', gap: 4, marginLeft: 32 }}>
                <div
                  style={{
                    width: 6,
                    height: 6,
                    backgroundColor: '#AAA',
                    borderRadius: '50%',
                    animation: 'bounce 1s infinite'
                  }}
                />
                <div
                  style={{
                    width: 6,
                    height: 6,
                    backgroundColor: '#AAA',
                    borderRadius: '50%',
                    animation: 'bounce 1s infinite',
                    animationDelay: '0.2s'
                  }}
                />
                <div
                  style={{
                    width: 6,
                    height: 6,
                    backgroundColor: '#AAA',
                    borderRadius: '50%',
                    animation: 'bounce 1s infinite',
                    animationDelay: '0.4s'
                  }}
                />
              </div>
            )}
          </div>

          {/* Input (sticky bottom) */}
          <div
            style={{
              padding: '16px',
              borderTop: '1px solid #E5E7EB',
              backgroundColor: '#FFF',
              borderBottomLeftRadius: 14,
              borderBottomRightRadius: 14
            }}
          >
            <div style={{ display: 'flex', gap: 12 }}>
              <input
                type="text"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSend()}
                placeholder="Type a message…"
                style={{
                  flex: 1,
                  padding: '12px 16px',
                  borderRadius: 9999,
                  border: '1px solid #D1D5DB',
                  outline: 'none',
                  fontSize: 14
                }}
              />
              <button
                onClick={handleSend}
                style={{
                  width: 40,
                  height: 40,
                  backgroundColor: '#990200',
                  border: 'none',
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer'
                }}
              >
                <svg
                  width="20"
                  height="20"
                  fill="none"
                  stroke="#FFF"
                  strokeWidth="2"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      )}
      <style>
        {`
          @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-4px); }
          }
        `}
      </style>
    </>
  );
}
