from datetime import datetime
from app import db

class ChatSession(db.Model):
    __tablename__ = 'chat_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(64), unique=True, nullable=False)  # Unique session identifier
    title = db.Column(db.String(255), nullable=True)  # Auto-generated title based on conversation
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=True)  # Optional link to dataset
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', backref=db.backref('chat_sessions', lazy='dynamic'))
    dataset = db.relationship('Dataset', backref=db.backref('chat_sessions', lazy='dynamic'))
    messages = db.relationship('ChatMessage', backref='session', lazy='dynamic', cascade='all, delete-orphan')
    
    def save(self):
        db.session.add(self)
        db.session.commit()
    
    def __repr__(self):
        return f'<ChatSession {self.session_id}>'