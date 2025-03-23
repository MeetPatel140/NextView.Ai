from flask import render_template, jsonify, request, current_app
from flask_login import login_required, current_user
from app.chatbot import bp
from app.models.chatbot import ChatSession
from app.models.chatbot_message import ChatMessage
from datetime import datetime

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
    
    # Create or get existing chat session
    chat_session = ChatSession.query.filter_by(user_id=current_user.id, is_active=True).first()
    if not chat_session:
        chat_session = ChatSession(user_id=current_user.id, title="New Chat", is_active=True)
        chat_session.save()
    
    # Save user message
    user_msg = ChatbotMessage(
        session_id=chat_session.id,
        content=user_message,
        role='user',
        created_at=datetime.utcnow()
    )
    user_msg.save()
    
    # This is a placeholder for actual AI processing
    # In a real implementation, this would call OpenAI or another AI service
    ai_response = "I'm sorry, the AI processing functionality is still being implemented. Please check back soon!"
    
    # Save AI response
    ai_msg = ChatbotMessage(
        session_id=chat_session.id,
        content=ai_response,
        role='assistant',
        created_at=datetime.utcnow()
    )
    ai_msg.save()
    
    return jsonify({
        'response': ai_response,
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