import json
from channels.generic.websocket import AsyncWebsocketConsumer

class PDFProcessingConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        self.group_name = f"pdf_processing_{self.session_id}"
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )  
        await self.accept()
        print("??? WebSocket connection established!")  # Should appear in your terminal

        # Send a test message when a client connects
        await self.send(text_data=json.dumps({
            "message": "WebSocket connected successfully! Please close notification to upload PDF cad drawing or Please wait till the autocad ai begins estimating process."
        }))

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )
        print("?? WebSocket disconnected!")

    async def notify_completion(self, event):
        await self.send(text_data=json.dumps({
            "message": event["message"],
            "total_page": event["total_page"],
            "cur_page": event["cur_page"],
            "pdf_url": event["pdf_url"],  # URL for PDF file
            "excel_url": event["excel_url"],  # URL for Excel file
            }))
        print("pdf processing completed!!!")