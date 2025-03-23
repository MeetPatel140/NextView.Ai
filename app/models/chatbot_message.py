from datetime import datetime
from app import db

class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_sessions.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)  # Message content
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Optional metadata for AI-generated responses
    message_metadata = db.Column(db.JSON, nullable=True)  # Store additional info like tokens, model used, etc.
    
    def __repr__(self):
        return f'<ChatMessage {self.id} - {self.role}>'