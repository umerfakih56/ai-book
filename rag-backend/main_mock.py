from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import uuid
import json

# Pydantic models
class ChatRequest(BaseModel):
    question: str
    session_id: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []

# Initialize FastAPI
app = FastAPI(title="Physical AI ChatBot API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "https://textbook-three.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Mock chat endpoint for testing"""
    try:
        print(f"Received chat request: {request.question}")
        
        # Mock responses based on keywords - improved logic
        question_lower = request.question.lower()
        
        if "what is physical ai" in question_lower:
            answer = """Physical AI refers to artificial intelligence systems that interact with the physical world through robots and other physical devices. It combines AI algorithms with robotics, sensors, and actuators to create intelligent systems that can perceive, reason, and act in real-world environments.

Key aspects of Physical AI include:
- Embodiment in physical form (robots, drones, etc.)
- Real-time perception and decision making
- Interaction with physical environments
- Integration of sensing, planning, and control

This field is crucial for applications like autonomous vehicles, industrial automation, and humanoid robots that can assist humans in daily tasks."""
        elif "how do i install ros 2" in question_lower or "install ros 2" in question_lower:
            answer = """For installing ROS 2, here are the basic steps:

1. Choose your ROS 2 distribution (Humble, Iron, etc.)
2. Set up your system (Ubuntu 22.04 recommended)
3. Add the ROS 2 apt repository:
   ```bash
   sudo apt update && sudo apt install curl gnupg lsb-release
   sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
   ```
4. Install ROS 2:
   ```bash
   sudo apt update
   sudo apt install ros-humble-desktop
   ```
5. Set up environment:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

Always check the official ROS 2 documentation for the most up-to-date installation instructions."""
        elif "explain nvidia isaac" in question_lower or "nvidia isaac sim" in question_lower:
            answer = """NVIDIA Isaac is a comprehensive platform for developing and deploying AI-powered robots. It includes simulation tools, AI frameworks, and hardware acceleration.

Key components:
- Isaac Sim: High-fidelity robotics simulation environment
- Isaac SDK: Development kit for robot applications
- Isaac AMR: Autonomous mobile robot platform
- Isaac Manipulator: Framework for robotic manipulation

The platform leverages NVIDIA's GPU computing power to enable advanced perception, navigation, and manipulation capabilities in robots."""
        elif "ros 2" in question_lower or "ros" in question_lower:
            answer = """ROS 2 (Robot Operating System 2) is the next-generation robotics middleware framework that provides the tools, libraries, and conventions needed to create complex robotic applications.

Key features of ROS 2:
- Real-time capabilities for time-critical applications
- Improved security and authentication
- Better support for multi-robot systems
- Cross-platform compatibility (Linux, Windows, macOS)
- Quality of Service (QoS) settings for reliable communication

ROS 2 is widely used in research and industry for developing autonomous systems, from small educational robots to large industrial applications."""
        elif "nvidia isaac" in question_lower or "isaac" in question_lower:
            answer = """NVIDIA Isaac is a comprehensive platform for developing and deploying AI-powered robots. It includes simulation tools, AI frameworks, and hardware acceleration.

Key components:
- Isaac Sim: High-fidelity robotics simulation environment
- Isaac SDK: Development kit for robot applications
- Isaac AMR: Autonomous mobile robot platform
- Isaac Manipulator: Framework for robotic manipulation

The platform leverages NVIDIA's GPU computing power to enable advanced perception, navigation, and manipulation capabilities in robots."""
        elif "install" in question_lower:
            answer = """For installing ROS 2, here are the basic steps:

1. Choose your ROS 2 distribution (Humble, Iron, etc.)
2. Set up your system (Ubuntu 22.04 recommended)
3. Add the ROS 2 apt repository:
   ```bash
   sudo apt update && sudo apt install curl gnupg lsb-release
   sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
   ```
4. Install ROS 2:
   ```bash
   sudo apt update
   sudo apt install ros-humble-desktop
   ```
5. Set up environment:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

Always check the official ROS 2 documentation for the most up-to-date installation instructions."""
        elif "physical ai" in question_lower:
            answer = """Physical AI refers to artificial intelligence systems that interact with the physical world through robots and other physical devices. It combines AI algorithms with robotics, sensors, and actuators to create intelligent systems that can perceive, reason, and act in real-world environments.

Key aspects of Physical AI include:
- Embodiment in physical form (robots, drones, etc.)
- Real-time perception and decision making
- Interaction with physical environments
- Integration of sensing, planning, and control

This field is crucial for applications like autonomous vehicles, industrial automation, and humanoid robots that can assist humans in daily tasks."""
        else:
            answer = f"""That's a great question about Physical AI and robotics! Based on your query about "{request.question}", I'd recommend exploring our comprehensive course materials which cover topics like:

- Physical AI fundamentals and applications
- ROS 2 programming and simulation
- NVIDIA Isaac platform and tools
- Humanoid robot design and control

For more detailed information, please check out our course documentation and tutorials. Is there a specific aspect of Physical AI or robotics you'd like to dive deeper into?"""
        
        print(f"Generated mock response: {answer[:100]}...")
        
        return ChatResponse(answer=answer, sources=[])
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/selected", response_model=ChatResponse)
async def chat_selected(request: ChatRequest):
    """Mock chat endpoint for selected text"""
    try:
        print(f"Received selected chat request: {request.question}")
        
        answer = f"""Based on the selected text context, here's a focused answer to your question: "{request.question}"

This response is tailored to the specific content you've selected from our Physical AI and Humanoid Robotics course materials. The selected text provides important context for understanding this topic within the broader field of physical AI and robotics.

For more comprehensive information, I recommend exploring the related sections in our course documentation."""
        
        return ChatResponse(answer=answer, sources=[], mode="selected_text")
        
    except Exception as e:
        print(f"Error in selected chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session (mock implementation)"""
    try:
        print(f"Getting history for session: {session_id}")
        # Return empty history for now
        return {"history": []}
    except Exception as e:
        print(f"Error in history endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
