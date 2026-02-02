"""Test conversational QA API"""
import requests
import json

BASE_URL = "http://localhost:9010"

# Create session
print("Creating session...")
resp = requests.post(f"{BASE_URL}/api/v1/chat/sessions", json={})
session_id = resp.json()["session_id"]
print(f"Session: {session_id[:12]}...")

# First question
print("\n1st question: Machine learning là gì?")
resp = requests.post(
    f"{BASE_URL}/api/v1/chat/",
    json={"session_id": session_id, "question": "Machine learning là gì?", "top_k": 3}
)
data = resp.json()
print(f"Answer: {data['answer'][:100]}...")
print(f"Messages: {data['conversation_length']}")

# Follow-up
print("\n2nd question (follow-up): Nó hoạt động thế nào?")
resp = requests.post(
    f"{BASE_URL}/api/v1/chat/",
    json={"session_id": session_id, "question": "Nó hoạt động thế nào?", "top_k": 3}
)
data = resp.json()
print(f"Answer: {data['answer'][:100]}...")
print(f"Reformulated: {data['reformulated_question'][:100]}...")
print(f"Used context: {data['used_context']}")
print(f"Messages: {data['conversation_length']}")

# Get history
print("\nGetting history...")
resp = requests.get(f"{BASE_URL}/api/v1/chat/sessions/{session_id}/history")
history = resp.json()["messages"]
print(f"Total messages: {len(history)}")
for i, msg in enumerate(history):
    print(f"  {i+1}. {msg['role']}: {msg['content'][:50]}...")

print("\nTest complete!")
