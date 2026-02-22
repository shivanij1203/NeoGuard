import asyncio
import json
import logging
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections for real-time pain score broadcasting."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.active_connections.remove(conn)


manager = ConnectionManager()


@router.websocket("/ws/monitor/{patient_id}")
async def monitor_patient(websocket: WebSocket, patient_id: int):
    await manager.connect(websocket)
    try:
        while True:
            # Receive frame/audio data from client or just keep-alive
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            elif msg.get("type") == "frame":
                # Process frame through ML pipeline (imported at runtime to avoid circular)
                from ml.scoring import process_frame_data
                result = await process_frame_data(msg.get("data"), patient_id)
                await websocket.send_json({
                    "type": "pain_update",
                    "patient_id": patient_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    **result,
                })
                # Broadcast to all dashboard connections
                await manager.broadcast({
                    "type": "pain_update",
                    "patient_id": patient_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    **result,
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.websocket("/ws/dashboard")
async def dashboard_feed(websocket: WebSocket):
    """Dashboard receives all patient updates via broadcast."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
