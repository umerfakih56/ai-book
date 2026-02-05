import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';

export default function Home() {
  return (
    <Layout
      title="Physical AI & Humanoid Robotics"
      description="The ultimate stack for intelligent robotics"
    >
      <div style={{ 
        background: "#000000",
        backgroundImage: `linear-gradient(rgba(255, 140, 0, 0.05) 1px, transparent 1px), 
                         linear-gradient(90deg, rgba(255, 140, 0, 0.05) 1px, transparent 1px)`,
        backgroundSize: '50px 50px',
        minHeight: "100vh",
        position: "relative",
        overflow: "hidden",
        color: "#ffffff",
        fontFamily: "'Segoe UI', Roboto, sans-serif"
      }}>
        
        {/* --- THE ROBOT (OR-B0) --- */}
        <div className="robot-container">
          <div className="robot-head">
            <div className="eye left"></div>
            <div className="eye right"></div>
          </div>
          <div className="robot-body"></div>
          <div className="robot-arm left"></div>
          <div className="robot-arm right"></div>
        </div>

        {/* Ambient Glows */}
        <div style={{
          position: "absolute", top: "-10%", right: "-5%", width: "500px", height: "500px",
          background: "radial-gradient(circle, rgba(255, 69, 0, 0.15) 0%, transparent 70%)",
          filter: "blur(80px)", zIndex: 1
        }} />

        {/* Hero Section */}
        <div style={{ position: "relative", zIndex: 2, padding: "6rem 2rem 4rem", textAlign: "center" }}>
          <div className="scanner-line"></div>
          
          <div style={{
            display: "inline-flex", alignItems: "center", gap: "10px",
            padding: "0.5rem 1.2rem", background: "rgba(255, 140, 0, 0.1)",
            border: "1px solid #ff8c00", borderRadius: "4px", marginBottom: "2rem"
          }}>
            <div className="status-dot"></div>
            <span style={{ fontSize: "0.8rem", fontWeight: "bold", color: "#ff8c00", textTransform: "uppercase", letterSpacing: "2px" }}>
              System Online: v2.0.26
            </span>
          </div>
          
          <h1 style={{ 
            fontSize: "clamp(2.5rem, 8vw, 4.5rem)", fontWeight: "900", marginBottom: "1rem",
            letterSpacing: "-1px", textShadow: "0 0 30px rgba(255, 69, 0, 0.3)"
          }}>
            PHYSICAL <span style={{ color: "#ff8c00" }}>AI</span> &<br />
            HUMANOID ROBOTICS
          </h1>
          
          <p style={{ 
            fontSize: "1.2rem", maxWidth: "800px", margin: "0 auto 3rem",
            color: "#aaaaaa", lineHeight: "1.8", fontWeight: "300"
          }}>
            Building the bridge between neural networks and physical actuators. 
            A deep-dive into the <span style={{color: "#fff"}}>NVIDIA Isaac</span> ecosystem, 
            <span style={{color: "#fff"}}>ROS 2</span> orchestration, and 
            <span style={{color: "#fff"}}>End-to-End</span> humanoid control.
          </p>
          
          <div style={{ display: "flex", gap: "1.5rem", justifyContent: "center", flexWrap: "wrap" }}>
            <Link to="/docs/intro" className="btn-primary">
              Initialize Learning_
            </Link>
            <Link to="/docs/setup" className="btn-secondary">
              Hardware Requirements
            </Link>
          </div>
        </div>

        {/* Course Journey Grid */}
        <div style={{ position: "relative", zIndex: 2, padding: "4rem 2rem", maxWidth: "1300px", margin: "0 auto" }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "1rem" }}>
            {[
              { id: "01", title: "Embodied AI Foundations", desc: "The physics of intelligence." },
              { id: "02", title: "Neural Orchestration", desc: "ROS 2 middleware & communication." },
              { id: "03", title: "Isaac Lab Simulation", desc: "Reinforcement learning at scale." },
              { id: "04", title: "VLA Architectures", desc: "Vision-Language-Action models." }
            ].map((card, i) => (
              <div key={i} className="tech-card">
                <div style={{ color: "#ff8c00", fontSize: "0.8rem", fontWeight: "bold", marginBottom: "1rem" }}>// MODULE_{card.id}</div>
                <h3 style={{ fontSize: "1.4rem", marginBottom: "0.5rem" }}>{card.title}</h3>
                <p style={{ color: "#777", fontSize: "0.9rem" }}>{card.desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* CSS FOR ENHANCEMENTS */}
        <style jsx>{`
          .btn-primary {
            background: #ff8c00;
            color: #000;
            padding: 1rem 2.5rem;
            border-radius: 4px;
            font-weight: 800;
            text-transform: uppercase;
            text-decoration: none;
            transition: 0.3s;
            box-shadow: 0 0 20px rgba(255, 140, 0, 0.4);
          }
          .btn-primary:hover { transform: translateY(-3px); box-shadow: 0 0 40px rgba(255, 140, 0, 0.6); }
          
          .btn-secondary {
            border: 1px solid #444;
            color: #fff;
            padding: 1rem 2.5rem;
            border-radius: 4px;
            text-decoration: none;
            transition: 0.3s;
          }
          .btn-secondary:hover { background: rgba(255,255,255,0.05); border-color: #ff8c00; }

          .tech-card {
            background: rgba(15, 15, 15, 0.8);
            border: 1px solid #222;
            padding: 2rem;
            backdrop-filter: blur(10px);
            transition: 0.3s;
            position: relative;
            overflow: hidden;
          }
          .tech-card:hover { border-color: #ff8c00; background: rgba(20, 20, 20, 0.9); }
          .tech-card:hover::after {
            content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 2px; background: #ff8c00;
          }

          /* --- CSS ROBOT ANIMATION --- */
          .robot-container {
            position: absolute;
            bottom: 50px;
            right: 50px;
            width: 80px;
            height: 100px;
            animation: float 4s ease-in-out infinite;
            z-index: 10;
          }
          .robot-head {
            width: 40px; height: 30px; background: #333; border: 2px solid #ff8c00; 
            border-radius: 8px; margin: 0 auto; position: relative;
            display: flex; justify-content: space-around; align-items: center;
          }
          .eye { width: 6px; height: 6px; background: #ff8c00; border-radius: 50%; box-shadow: 0 0 10px #ff8c00; }
          .robot-body {
            width: 50px; height: 40px; background: #222; border: 2px solid #ff8c00;
            border-radius: 4px; margin: 5px auto;
          }
          .robot-arm {
            position: absolute; width: 10px; height: 30px; background: #ff8c00; top: 40px; border-radius: 5px;
          }
          .arm.left { left: 0; }
          .arm.right { right: 0; }

          @keyframes float {
            0%, 100% { transform: translateY(0) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(5deg); }
          }

          .status-dot {
            width: 8px; height: 8px; background: #ff8c00; border-radius: 50%;
            animation: blink 1s infinite;
          }
          @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

          .scanner-line {
            position: absolute; top: 0; left: 0; width: 100%; height: 2px;
            background: linear-gradient(90deg, transparent, #ff8c00, transparent);
            opacity: 0.2; animation: scan 8s linear infinite;
          }
          @keyframes scan { 0% { top: 0%; } 100% { top: 100%; } }
        `}</style>
      </div>
    </Layout>
  );
}