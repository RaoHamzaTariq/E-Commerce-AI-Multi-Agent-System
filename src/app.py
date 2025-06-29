# app.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
import uuid
from typing import Dict, List
from src.main import chatbot
from datetime import datetime

app = FastAPI()

# Enable logging
logging.basicConfig(level=logging.INFO)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # List of origins that are allowed to make cross-origin requests
    allow_credentials=True, # Allow cookies to be sent with requests
    allow_methods=["*"],    # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],    # Allow all headers in the request
)   

@app.websocket("/ws/chat")
async def websocket_chat(websocket:WebSocket):
    await websocket.accept()

    logging.info("Client connected.")

    # Memory store per session
    memory = []

    try:
        while True:
            # Receive message from client
            user_input = await websocket.receive_text()
            logging.info(f"User: {user_input}")

            # Append user message to memory
            memory.append({"role": "user", "content": user_input})

            # Pass conversation history to the assistant
            result = await chatbot(memory)

            # Add assistant response to memory
            memory.append({"role": "assistant", "content": result.final_output})

            # Send response to client
            await websocket.send_text(result.final_output)
            logging.info(f"Assistant: {result.final_output}")

    except WebSocketDisconnect:
        logging.info("Client disconnected.")

        

@app.get("/")
async def root():
    return {"message": "Welcome to the Nike Ecommerce Chatbot WebSocket Service"}
