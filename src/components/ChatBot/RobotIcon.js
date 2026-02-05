import React from 'react';
import './RobotIcon.css';

const RobotIcon = ({ size = 32, className = '' }) => {
  return (
    <div 
      className={`robot-icon ${className}`}
      style={{ width: size, height: size }}
    >
      <svg
        width={size}
        height={size}
        viewBox="0 0 32 32"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Robot Head */}
        <rect
          x="4"
          y="8"
          width="24"
          height="16"
          rx="4"
          fill="url(#headGradient)"
          stroke="url(#borderGradient)"
          strokeWidth="0.5"
        />
        
        {/* Antenna */}
        <line
          x1="16"
          y1="8"
          x2="16"
          y2="4"
          stroke="url(#borderGradient)"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
        
        {/* Antenna LED */}
        <circle
          cx="16"
          cy="3"
          r="1.5"
          fill="#ff4757"
          className="antenna-led"
        >
          <animate
            attributeName="opacity"
            values="0.3;1;0.3"
            dur="2s"
            repeatCount="indefinite"
          />
        </circle>
        
        {/* Left Eye */}
        <g className="left-eye">
          <circle
            cx="11"
            cy="14"
            r="2.5"
            fill="#ffffff"
            stroke="url(#borderGradient)"
            strokeWidth="0.5"
          />
          <circle
            cx="11"
            cy="14"
            r="1.5"
            fill="#2c3e50"
            className="pupil"
          >
            <animate
              attributeName="r"
              values="1.5;0.8;1.5"
              dur="4s"
              repeatCount="indefinite"
            />
          </circle>
        </g>
        
        {/* Right Eye */}
        <g className="right-eye">
          <circle
            cx="21"
            cy="14"
            r="2.5"
            fill="#ffffff"
            stroke="url(#borderGradient)"
            strokeWidth="0.5"
          />
          <circle
            cx="21"
            cy="14"
            r="1.5"
            fill="#2c3e50"
            className="pupil"
          >
            <animate
              attributeName="r"
              values="1.5;0.8;1.5"
              dur="4s"
              begin="0.5s"
              repeatCount="indefinite"
            />
          </circle>
        </g>
        
        {/* Mouth - Friendly Smile */}
        <path
          d="M 12 18 Q 16 20 20 18"
          stroke="#ffffff"
          strokeWidth="1.5"
          strokeLinecap="round"
          fill="none"
          className="mouth"
        >
          <animate
            attributeName="d"
            values="M 12 18 Q 16 20 20 18;M 12 18 Q 16 19 20 18;M 12 18 Q 16 20 20 18"
            dur="3s"
            repeatCount="indefinite"
          />
        </path>
        
        {/* Gradients */}
        <defs>
          <linearGradient id="headGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#667eea" />
            <stop offset="100%" stopColor="#764ba2" />
          </linearGradient>
          
          <linearGradient id="borderGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#5a6fd8" />
            <stop offset="100%" stopColor="#6a4190" />
          </linearGradient>
          
          {/* Glow effect for LED */}
          <radialGradient id="ledGlow">
            <stop offset="0%" stopColor="#ff4757" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#ff4757" stopOpacity="0" />
          </radialGradient>
        </defs>
        
        {/* LED Glow Effect */}
        <circle
          cx="16"
          cy="3"
          r="3"
          fill="url(#ledGlow)"
          className="led-glow"
        >
          <animate
            attributeName="opacity"
            values="0;0.6;0"
            dur="2s"
            repeatCount="indefinite"
          />
        </circle>
      </svg>
    </div>
  );
};

export default RobotIcon;
