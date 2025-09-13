import json
from channels.generic.websocket import AsyncWebsocketConsumer

class PDFProcessingConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.group_name = "pdf_processing"
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )  
        await self.accept()
        print("🛠️ WebSocket connection established!")  # Should appear in your terminal

        # Send a test message when a client connects
        await self.send(text_data=json.dumps({
            "message": "WebSocket connected successfully!"
        }))

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )
        print("⚠️ WebSocket disconnected!")

    async def notify_completion(self, event):
        await self.send(text_data=json.dumps({
            "message": "Processing complete",
            "pdf_url": event["pdf_url"],  # URL for PDF file
            "excel_url": event["excel_url"],  # URL for Excel file
            }))
        print("pdf processing completed!!!")