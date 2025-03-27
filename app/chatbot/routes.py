from flask import render_template, jsonify, request, current_app
from flask_login import login_required, current_user
from app.chatbot import bp
from app.models.chatbot import ChatSession
from app.models.chatbot_message import ChatMessage
from app.chatbot.ai_service import AIService
from datetime import datetime
import json

@bp.route('/chat')
@login_required
def chat():
    """Display the chat interface"""
    return render_template('chat/chat.html', title='AI Assistant')

@bp.route('/api/send', methods=['POST'])
@login_required
def send_message():
    """Process a message sent to the chatbot"""
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    
    user_message = data['message']
    dataset_id = data.get('dataset_id')  # Optional dataset ID if specified
    
    # Create or get existing chat session
    chat_session = ChatSession.query.filter_by(user_id=current_user.id, is_active=True).first()
    if not chat_session:
        chat_session = ChatSession(user_id=current_user.id, title="New Chat", is_active=True)
        chat_session.save()
    
    # Save user message
    user_msg = ChatbotMessage(
    user_msg = ChatbotMessage(
        session_id=chat_session.id,
        content=user_message,
        role='user',
        created_at=datetime.utcnow()
    )
    user_msg.save()
    
    # Get image URL if provided
    image_url = data.get('image_url')
    
    # Get chat history for context
    chat_history = ChatMessage.query.filter_by(session_id=chat_session.id).order_by(ChatMessage.created_at).all()
    messages = [{
        'role': msg.role,
        'content': msg.content
    } for msg in chat_history]
    
    # Add current user message
    messages.append({
        'role': 'user',
        'content': user_message
    })
    
    # Get AI response using OpenRouter API
    ai_service = AIService()
    ai_response = ai_service.create_chat_completion(messages, image_url)
    
    # Save AI response
    ai_msg = ChatbotMessage(
    ai_msg = ChatbotMessage(
        session_id=chat_session.id,
        content=ai_response,
        role='assistant',
        created_at=datetime.utcnow()
        created_at=datetime.utcnow()
    )
    ai_msg.save()
    
    return jsonify({
        'response': ai_response,
        'metadata': response_metadata,
        'timestamp': datetime.utcnow().isoformat()
    })

@bp.route('/api/history')
@login_required
def get_history():
    """Get chat history for the current user"""
    chat_session = ChatSession.query.filter_by(user_id=current_user.id, is_active=True).first()
    if not chat_session:
        return jsonify({'messages': []})
    
    messages = ChatbotMessage.query.filter_by(session_id=chat_session.id).order_by(ChatbotMessage.created_at).all()
    message_list = [{
        'content': msg.content,
        'role': msg.role,
        'timestamp': msg.created_at.isoformat()
    } for msg in messages]
    
    return jsonify({'messages': message_list})