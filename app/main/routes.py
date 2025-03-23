from flask import render_template, flash, redirect, url_for, request, jsonify, current_app, send_file
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
import pandas as pd
import json
from datetime import datetime
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
import io
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from app import db, celery
from app.main import bp
from app.models.dataset import Dataset, Visualization
from app.models.chatbot import ChatSession
from app.models.chatbot_message import ChatMessage

# Helper functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@celery.task(bind=True)
def process_dataset(self, dataset_id):
    """Process uploaded dataset and extract metadata"""
    dataset = Dataset.query.get(dataset_id)
    if not dataset:
        return False
    
    try:
        # Read the Excel file
        df = pd.read_excel(dataset.file_path)
        
        # Extract metadata
        dataset.row_count = len(df)
        dataset.column_count = len(df.columns)
        
        # Store column information
        columns_info = []
        for col in df.columns:
            col_type = str(df[col].dtype)
            unique_values = df[col].nunique()
            missing_values = df[col].isna().sum()
            
            columns_info.append({
                'name': col,
                'type': col_type,
                'unique_values': int(unique_values),
                'missing_values': int(missing_values)
            })
        
        # Store metadata as JSON
        dataset.dataset_metadata = {
            'columns': columns_info,
            'processed_at': datetime.utcnow().isoformat()
        }
        
        dataset.is_processed = True
        db.session.commit()
        return True
    except Exception as e:
        current_app.logger.error(f"Error processing dataset {dataset_id}: {str(e)}")
        return False

# Main routes
@bp.route('/')
@bp.route('/index')
def index():
    """Landing page"""
    return render_template('index.html', title='Welcome to NextView.AI')

@bp.route('/dashboard')
@login_required
def dashboard():
    """User dashboard"""
    datasets = Dataset.query.filter_by(user_id=current_user.id).all()
    visualizations = Visualization.query.filter_by(user_id=current_user.id).all()
    chat_sessions = ChatSession.query.filter_by(user_id=current_user.id).all()
    
    return render_template('dashboard.html', 
                           title='Dashboard',
                           datasets=datasets,
                           visualizations=visualizations,
                           chat_sessions=chat_sessions)

# Dataset routes
@bp.route('/datasets', methods=['GET', 'POST'])
@login_required
def datasets():
    """List and upload datasets"""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Create unique filename to prevent overwriting
            unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save the file
            file.save(file_path)
            
            # Create dataset record
            dataset = Dataset(
                name=request.form.get('name', filename),
                description=request.form.get('description', ''),
                file_path=file_path,
                file_type='xlsx',
                user_id=current_user.id
            )
            
            db.session.add(dataset)
            db.session.commit()
            
            # Process dataset asynchronously
            process_dataset.delay(dataset.id)
            
            flash('Dataset uploaded successfully!', 'success')
            return redirect(url_for('main.dataset_detail', dataset_id=dataset.id))
    
    # GET request - show datasets
    datasets = Dataset.query.filter_by(user_id=current_user.id).all()
    return render_template('datasets/index.html', title='My Datasets', datasets=datasets)

@bp.route('/datasets/<int:dataset_id>')
@login_required
def dataset_detail(dataset_id):
    """Show dataset details"""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Check if user owns this dataset
    if dataset.user_id != current_user.id:
        flash('You do not have permission to view this dataset', 'danger')
        return redirect(url_for('main.datasets'))
    
    # If dataset is processed, load preview data
    preview_data = None
    columns = []
    if dataset.is_processed:
        try:
            df = pd.read_excel(dataset.file_path)
            preview_data = df.head(10).to_dict('records')
            columns = df.columns.tolist()
        except Exception as e:
            flash(f'Error loading dataset preview: {str(e)}', 'danger')
    
    return render_template('datasets/detail.html', 
                           title=dataset.name,
                           dataset=dataset,
                           preview_data=preview_data,
                           columns=columns)

# Visualization routes
@bp.route('/visualizations', methods=['GET'])
@login_required
def visualizations():
    """List visualizations"""
    visualizations = Visualization.query.filter_by(user_id=current_user.id).all()
    return render_template('visualizations/index.html', 
                           title='My Visualizations', 
                           visualizations=visualizations)

@bp.route('/visualizations/create', methods=['GET', 'POST'])
@login_required
def create_visualization():
    """Create a new visualization"""
    datasets = Dataset.query.filter_by(user_id=current_user.id, is_processed=True).all()
    
    if request.method == 'POST':
        dataset_id = request.form.get('dataset_id')
        name = request.form.get('name')
        description = request.form.get('description', '')
        chart_type = request.form.get('chart_type')
        
        # Get chart configuration from form
        config = {
            'x_axis': request.form.get('x_axis'),
            'y_axis': request.form.get('y_axis'),
            'aggregation': request.form.get('aggregation', 'sum'),
            'filters': request.form.getlist('filters[]'),
            'colors': request.form.get('colors', 'default')
        }
        
        # Create visualization
        visualization = Visualization(
            name=name,
            description=description,
            chart_type=chart_type,
            config=config,
            dataset_id=dataset_id,
            user_id=current_user.id
        )
        
        db.session.add(visualization)
        db.session.commit()
        
        flash('Visualization created successfully!', 'success')
        return redirect(url_for('main.visualization_detail', visualization_id=visualization.id))
    
    return render_template('visualizations/create.html', 
                           title='Create Visualization',
                           datasets=datasets)

@bp.route('/visualizations/<int:visualization_id>')
@login_required
def visualization_detail(visualization_id):
    """Show visualization details and render chart"""
    visualization = Visualization.query.get_or_404(visualization_id)
    
    # Check if user owns this visualization
    if visualization.user_id != current_user.id:
        flash('You do not have permission to view this visualization', 'danger')
        return redirect(url_for('main.visualizations'))
    
    return render_template('visualizations/detail.html',
                           title=visualization.name,
                           visualization=visualization)

@bp.route('/api/visualizations/<int:visualization_id>/data')
@login_required
def visualization_data(visualization_id):
    """API endpoint to get visualization data for charts"""
    visualization = Visualization.query.get_or_404(visualization_id)
    
    # Check if user owns this visualization
    if visualization.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        # Get dataset
        dataset = Dataset.query.get(visualization.dataset_id)
        df = pd.read_excel(dataset.file_path)
        
        # Apply configuration
        config = visualization.config
        x_axis = config.get('x_axis')
        y_axis = config.get('y_axis')
        aggregation = config.get('aggregation', 'sum')
        
        # Apply filters if any
        filters = config.get('filters', [])
        for filter_config in filters:
            if 'column' in filter_config and 'value' in filter_config and 'operator' in filter_config:
                column = filter_config['column']
                value = filter_config['value']
                operator = filter_config['operator']
                
                if operator == 'equals':
                    df = df[df[column] == value]
                elif operator == 'not_equals':
                    df = df[df[column] != value]
                elif operator == 'greater_than':
                    df = df[df[column] > float(value)]
                elif operator == 'less_than':
                    df = df[df[column] < float(value)]
        
        # Perform aggregation
        if aggregation == 'sum':
            result_df = df.groupby(x_axis)[y_axis].sum().reset_index()
        elif aggregation == 'avg':
            result_df = df.groupby(x_axis)[y_axis].mean().reset_index()
        elif aggregation == 'count':
            result_df = df.groupby(x_axis)[y_axis].count().reset_index()
        elif aggregation == 'max':
            result_df = df.groupby(x_axis)[y_axis].max().reset_index()
        elif aggregation == 'min':
            result_df = df.groupby(x_axis)[y_axis].min().reset_index()
        else:
            result_df = df.groupby(x_axis)[y_axis].sum().reset_index()
        
        # Convert to chart data format
        chart_data = {
            'labels': result_df[x_axis].tolist(),
            'datasets': [{
                'label': y_axis,
                'data': result_df[y_axis].tolist()
            }]
        }
        
        return jsonify(chart_data)
    except Exception as e:
        current_app.logger.error(f"Error generating visualization data: {str(e)}")
        return jsonify({'error': str(e)}), 500


    # Check if user owns this visualization
    if visualization.user_id != current_user.id:
        flash('You do not have permission to export this visualization', 'danger')
        return redirect(url_for('main.visualizations'))
    
    try:
        # Get dataset
        dataset = Dataset.query.get(visualization.dataset_id)
        df = pd.read_excel(dataset.file_path)
        
        # Create a BytesIO buffer for the PDF
        buffer = io.BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        
        # Add title
        styles = getSampleStyleSheet()
        title = Paragraph(f"<b>{visualization.name}</b>", styles['Title'])
        elements.append(title)
        
        # Add description if available
        if visualization.description:
            description = Paragraph(visualization.description, styles['Normal'])
            elements.append(description)
        
        elements.append(Spacer(1, 12))
        
        # Add dataset info
        dataset_info = Paragraph(f"<b>Dataset:</b> {dataset.name}", styles['Normal'])
        elements.append(dataset_info)
        elements.append(Spacer(1, 12))
        
        # Generate chart image
        config = visualization.config
        x_axis = config.get('x_axis')
        y_axis = config.get('y_axis')
        aggregation = config.get('aggregation', 'sum')
        
        # Apply aggregation
        if aggregation == 'sum':
            result_df = df.groupby(x_axis)[y_axis].sum().reset_index()
        elif aggregation == 'avg':
            result_df = df.groupby(x_axis)[y_axis].mean().reset_index()
        elif aggregation == 'count':
            result_df = df.groupby(x_axis)[y_axis].count().reset_index()
        elif aggregation == 'max':
            result_df = df.groupby(x_axis)[y_axis].max().reset_index()
        elif aggregation == 'min':
            result_df = df.groupby(x_axis)[y_axis].min().reset_index()
        else:
            result_df = df.groupby(x_axis)[y_axis].sum().reset_index()
        
        # Create chart based on chart type
        plt.figure(figsize=(8, 6))
        chart_type = visualization.chart_type
        
        if chart_type == 'bar':
            plt.bar(result_df[x_axis], result_df[y_axis])
        elif chart_type == 'line':
            plt.plot(result_df[x_axis], result_df[y_axis])
        elif chart_type == 'pie':
            plt.pie(result_df[y_axis], labels=result_df[x_axis], autopct='%1.1f%%')
        elif chart_type == 'scatter':
            plt.scatter(result_df[x_axis], result_df[y_axis])
        else:  # Default to bar
            plt.bar(result_df[x_axis], result_df[y_axis])
        
        plt.title(f"{visualization.name}")
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.tight_layout()
        
        # Save chart to BytesIO
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        plt.close()
        
        # Add chart to PDF
        img = Image(img_buffer, width=450, height=300)
        elements.append(img)
        elements.append(Spacer(1, 12))
        
        # Add data table
        table_data = [[x_axis, y_axis]]
        for _, row in result_df.iterrows():
            table_data.append([str(row[x_axis]), str(row[y_axis])])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        
        # Add timestamp
        elements.append(Spacer(1, 20))
        timestamp = Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
        elements.append(timestamp)
        
        # Build the PDF
        doc.build(elements)
        
        # Get the PDF content
        pdf_content = buffer.getvalue()
        buffer.close()
        
        # Create response
        response = make_response(pdf_content)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename={visualization.name}.pdf'
        
        return response
    except Exception as e:
        flash(f'Error exporting visualization: {str(e)}', 'danger')
        return redirect(url_for('main.visualizations'))

# Chatbot routes
@bp.route('/chat')
@login_required
def chat():
    """List chat sessions"""
    chat_sessions = ChatSession.query.filter_by(user_id=current_user.id).order_by(ChatSession.updated_at.desc()).all()
    return render_template('chat/index.html', title='AI Chat', chat_sessions=chat_sessions)

@bp.route('/chat/new', methods=['GET', 'POST'])
@login_required
def new_chat():
    """Create a new chat session"""
    if request.method == 'POST':
        title = request.form.get('title', '')
        dataset_id = request.form.get('dataset_id')
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Create chat session
        chat_session = ChatSession(
            session_id=session_id,
            title=title,
            user_id=current_user.id,
            dataset_id=dataset_id if dataset_id else None
        )
        
        db.session.add(chat_session)
        db.session.commit()
        
        # Add welcome message
        welcome_message = ChatMessage(
            session_id=chat_session.id,
            content="Hello! I'm your AI assistant. How can I help you today?",
            role='assistant'
        )
        
        db.session.add(welcome_message)
        db.session.commit()
        
        return redirect(url_for('main.chat_session', session_id=session_id))
    
    # GET request - show form
    datasets = Dataset.query.filter_by(user_id=current_user.id, is_processed=True).all()
    return render_template('chat/new.html', title='New Chat', datasets=datasets)

@bp.route('/chat/<string:session_id>')
@login_required
def chat_session(session_id):
    """Show chat session"""
    session = ChatSession.query.filter_by(session_id=session_id).first_or_404()
    
    # Check if user owns this session
    if session.user_id != current_user.id:
        flash('You do not have permission to view this chat session', 'danger')
        return redirect(url_for('main.chat'))
    
    # Get messages
    messages = ChatMessage.query.filter_by(session_id=session.id).order_by(ChatMessage.created_at).all()
    
    return render_template('chat/session.html', 
                           title=session.title or 'Chat',
                           session=session,
                           messages=messages)

@bp.route('/api/chat/<string:session_id>/message', methods=['POST'])
@login_required
def send_message(session_id):
    """API endpoint to send a message and get AI response"""
    session = ChatSession.query.filter_by(session_id=session_id).first_or_404()
    
    # Check if user owns this session
    if session.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    # Get message from request
    message_content = request.form.get('message', '')
    if not message_content:
        return jsonify({'error': 'Message cannot be empty'}), 400
    
    # Save user message
    user_message = ChatMessage(
        session_id=session.id,
        content=message_content,
        role='user'
    )
    db.session.add(user_message)
    
    # Update session timestamp
    session.updated_at = datetime.utcnow()
    
    # Generate AI response
    try:
        # If session is linked to a dataset, include dataset context
        dataset_context = None
        if session.dataset_id:
            dataset = Dataset.query.get(session.dataset_id)
            if dataset and dataset.is_processed:
                # Load dataset preview for context
                df = pd.read_excel(dataset.file_path)
                dataset_context = {
                    'name': dataset.name,
                    'columns': df.columns.tolist(),
                    'preview': df.head(5).to_dict('records'),
                    'metadata': dataset.dataset_metadata
                }
        
        # Get previous messages for context (limit to last 10)
        previous_messages = ChatMessage.query.filter_by(session_id=session.id)\
            .order_by(ChatMessage.created_at.desc())\
            .limit(10)\
            .all()
        previous_messages.reverse()  # Chronological order
        
        # Format messages for AI context
        message_history = [{
            'role': msg.role,
            'content': msg.content
        } for msg in previous_messages]
        
        # Add current message
        message_history.append({
            'role': 'user',
            'content': message_content
        })
        
        # Prepare prompt with dataset context if available
        system_prompt = "You are an AI assistant for data analysis."
        if dataset_context:
            system_prompt += f" You are analyzing a dataset named '{dataset_context['name']}' with columns: {', '.join(dataset_context['columns'])}."
        
        # TODO: Replace with actual OpenAI or other LLM integration
        # For now, generate a simple response based on the message
        if "hello" in message_content.lower() or "hi" in message_content.lower():
            ai_response = "Hello! How can I help you with your data analysis today?"
        elif "dataset" in message_content.lower() or "data" in message_content.lower():
            if dataset_context:
                ai_response = f"I'm analyzing the dataset '{dataset_context['name']}'. It has {len(dataset_context['columns'])} columns. What would you like to know about it?"
            else:
                ai_response = "You haven't linked a dataset to this chat. Would you like to start a new chat with a dataset?"
        elif "visualize" in message_content.lower() or "chart" in message_content.lower() or "graph" in message_content.lower():
            ai_response = "I can help you create visualizations. You can use the Visualizations section to create charts based on your dataset."
        elif "analyze" in message_content.lower() or "statistics" in message_content.lower():
            ai_response = "I can help you analyze your data. What specific insights are you looking for?"
        else:
            ai_response = "I'm here to help with your data analysis needs. You can ask me about your datasets, visualizations, or general data questions."
        
        # Save AI response
        assistant_message = ChatMessage(
            session_id=session.id,
            content=ai_response,
            role='assistant',
            message_metadata={
                'generated_at': datetime.utcnow().isoformat(),
                'model': 'simple-rule-based'  # Replace with actual model info when integrated
            }
        )
        db.session.add(assistant_message)
        db.session.commit()
        
        # If no title yet, generate one based on first user message
        if not session.title and len(previous_messages) <= 1:
            session.title = message_content[:50] + ('...' if len(message_content) > 50 else '')
            db.session.commit()
        
        return jsonify({'response': ai_response})
    except Exception as e:
        current_app.logger.error(f"Error generating AI response: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/chat/<string:session_id>/delete')
@login_required
def delete_chat(session_id):
    """Delete a chat session"""
    session = ChatSession.query.filter_by(session_id=session_id).first_or_404()
    
    # Check if user owns this session
    if session.user_id != current_user.id:
        flash('You do not have permission to delete this chat session', 'danger')
        return redirect(url_for('main.chat'))
    
    # Delete session and all messages (cascade)
    db.session.delete(session)
    db.session.commit()
    
    flash('Chat session deleted successfully', 'success')
    return redirect(url_for('main.chat'))

@bp.route('/profile')
@login_required
def profile():
    """User profile page"""
    datasets = Dataset.query.filter_by(user_id=current_user.id).all()
    visualizations = Visualization.query.filter_by(user_id=current_user.id).all()
    chat_sessions = ChatSession.query.filter_by(user_id=current_user.id).all()
    
    return render_template('profile.html', 
                           title='My Profile',
                           datasets=datasets,
                           visualizations=visualizations,
                           chat_sessions=chat_sessions)


    # Check if user owns this visualization
    if visualization.user_id != current_user.id:
        flash('You do not have permission to export this visualization', 'danger')
        return redirect(url_for('main.visualizations'))
    
    try:
        # Get dataset
        dataset = Dataset.query.get(visualization.dataset_id)
        df = pd.read_excel(dataset.file_path)
        
        # Create a BytesIO buffer for the PDF
        buffer = io.BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        
        # Add title
        styles = getSampleStyleSheet()
        title = Paragraph(f"<b>{visualization.name}</b>", styles['Title'])
        elements.append(title)
        
        # Add description if available
        if visualization.description:
            description = Paragraph(visualization.description, styles['Normal'])
            elements.append(description)
        
        elements.append(Spacer(1, 12))
        
        # Add dataset info
        dataset_info = Paragraph(f"<b>Dataset:</b> {dataset.name}", styles['Normal'])
        elements.append(dataset_info)
        elements.append(Spacer(1, 12))
        
        # Generate chart image
        config = visualization.config
        x_axis = config.get('x_axis')
        y_axis = config.get('y_axis')
        aggregation = config.get('aggregation', 'sum')
        
        # Apply aggregation
        if aggregation == 'sum':
            result_df = df.groupby(x_axis)[y_axis].sum().reset_index()
        elif aggregation == 'avg':
            result_df = df.groupby(x_axis)[y_axis].mean().reset_index()
        elif aggregation == 'count':
            result_df = df.groupby(x_axis)[y_axis].count().reset_index()
        elif aggregation == 'max':
            result_df = df.groupby(x_axis)[y_axis].max().reset_index()
        elif aggregation == 'min':
            result_df = df.groupby(x_axis)[y_axis].min().reset_index()
        else:
            result_df = df.groupby(x_axis)[y_axis].sum().reset_index()
        
        # Create chart based on chart type
        plt.figure(figsize=(8, 6))
        chart_type = visualization.chart_type
        
        if chart_type == 'bar':
            plt.bar(result_df[x_axis], result_df[y_axis])
        elif chart_type == 'line':
            plt.plot(result_df[x_axis], result_df[y_axis])
        elif chart_type == 'pie':
            plt.pie(result_df[y_axis], labels=result_df[x_axis], autopct='%1.1f%%')
        elif chart_type == 'scatter':
            plt.scatter(result_df[x_axis], result_df[y_axis])
        else:  # Default to bar
            plt.bar(result_df[x_axis], result_df[y_axis])
        
        plt.title(f"{visualization.name}")
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.tight_layout()
        
        # Save chart to BytesIO
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        plt.close()
        
        # Add chart to PDF
        img = Image(img_buffer, width=450, height=300)
        elements.append(img)
        elements.append(Spacer(1, 12))
        
        # Add data table
        table_data = [[x_axis, y_axis]]
        for _, row in result_df.iterrows():
            table_data.append([str(row[x_axis]), str(row[y_axis])])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        
        # Add timestamp
        elements.append(Spacer(1, 20))
        timestamp = Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
        elements.append(timestamp)
        
        # Build the PDF
        doc.build(elements)
        
        # Get the PDF content
        pdf_content = buffer.getvalue()
        buffer.close()
        
        # Create response
        response = make_response(pdf_content)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename={visualization.name}.pdf'
        
        return response
    except Exception as e:
        flash(f'Error exporting visualization: {str(e)}', 'danger')
        return redirect(url_for('main.visualizations'))