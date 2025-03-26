from flask import render_template, jsonify, request, current_app
from flask_login import login_required, current_user
from app.chatbot import bp
from app.models.chatbot import ChatSession
from app.models.chatbot_message import ChatMessage
from app.chatbot.nlp_integration import ChatbotNLPIntegration
from app.models.dataset import Dataset
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
    user_msg = ChatMessage(
        session_id=chat_session.id,
        content=user_message,
        role='user',
        created_at=datetime.utcnow()
    )
    user_msg.save()
    
    # Check if this is a data visualization/analysis query or a report query
    is_data_query = ChatbotNLPIntegration.detect_data_query(user_message)
    is_report_query = ChatbotNLPIntegration.detect_report_query(user_message)
    
    if is_report_query:
        # Process as a comprehensive report request
        report_result = ChatbotNLPIntegration.generate_comprehensive_report(user_message, dataset_id)
        
        if report_result['success']:
            # Create a rich response with multiple visualizations
            ai_response = report_result['html']
            
            # Add metadata for frontend processing
            response_metadata = {
                'type': 'comprehensive_report',
                'visualizations': report_result['visualizations'],
                'insights': report_result['insights'],
                'dataset_name': report_result.get('dataset_name', 'Unknown dataset')
            }
        else:
            # If report generation failed, return the error
            ai_response = f"<p>I couldn't generate the comprehensive report: {report_result['error']}</p>"
            response_metadata = {'type': 'error'}
    elif is_data_query:
        # Process with advanced NLP processor
        nlp_result = ChatbotNLPIntegration.process_data_query(user_message, dataset_id)
        
        if nlp_result['success']:
            # Create a rich response with visualization
            ai_response = nlp_result['html']
            
            # Add metadata for frontend processing
            response_metadata = {
                'type': 'data_visualization',
                'visualization': nlp_result['visualization'],
                'insights': nlp_result['insights'],
                'dataset_name': nlp_result.get('dataset_name', 'Unknown dataset')
            }
        else:
            # If NLP processing failed, return the error
            ai_response = f"<p>I couldn't process your data query: {nlp_result['error']}</p>"
            response_metadata = {'type': 'error'}
    else:
        # This is a placeholder for actual AI processing for non-data queries
        # In a real implementation, this would call OpenAI or another AI service
        ai_response = "I'm sorry, the AI processing functionality is still being implemented. Please check back soon!"
        response_metadata = {'type': 'text'}
    
    # Save AI response
    ai_msg = ChatMessage(
        session_id=chat_session.id,
        content=ai_response,
        role='assistant',
        created_at=datetime.utcnow(),
        message_metadata=response_metadata
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