"""
Simple text-mode test (no microphone required).
Requires AICREDITS_API_KEY in .env
Run: python -m tests.test_agent_text
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.voice_agent import VoiceAgent
from src.utils.logger import setup_logger

setup_logger("INFO")

def main():
    agent = VoiceAgent()
    
    test_inputs = [
        "Hi, I'm interested in your AI voice agent.",
        "What is the pricing?",
        "Can it speak Hindi?",
        "I want to book a demo for tomorrow at 3 PM. My name is Rahul and phone is +919876543210.",
    ]
    
    print("\n=== Running automated text test (AICredits) ===\n")
    for text in test_inputs:
        print(f"User: {text}")
        response = agent.process_text(text)
        print(f"Agent: {response}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()
