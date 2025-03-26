from flask import render_template, flash, redirect, url_for, request, jsonify, current_app, send_file, make_response
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
import logging
import random
from flask_wtf.csrf import validate_csrf
from app.nlp.processor import NLPProcessor

# Helper functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@celery.task(name='app.main.routes.process_dataset', bind=True)
def process_dataset(self, dataset_id):
    """Process uploaded dataset and extract metadata"""
    try:
        # Import required modules inside the task to avoid circular imports
        import os
        import pandas as pd
        from datetime import datetime
        from app import db
        from app.models.dataset import Dataset
        from flask import current_app
        import logging
        
        # Set up basic logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info(f"Starting to process dataset with ID: {dataset_id}")
        
        # Get dataset 
        dataset = Dataset.query.get(dataset_id)
        if not dataset:
            logger.error(f"Dataset with ID {dataset_id} not found")
            return False
        
        # Check if file exists
        if not os.path.exists(dataset.file_path):
            logger.error(f"Dataset file not found at {dataset.file_path}")
            dataset.is_processed = False
            db.session.commit()
            return False
        # Read the Excel file
        logger.info(f"Reading Excel file: {dataset.file_path}")
        df = pd.read_excel(dataset.file_path)
        
        # Extract metadata
        logger.info(f"Extracting metadata from dataset with {len(df)} rows and {len(df.columns)} columns")
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
        
        # Mark as processed and save
        dataset.is_processed = True
        db.session.commit()
        logger.info(f"Successfully processed dataset {dataset_id}")
        return True
    except Exception as e:
        logger.error(f"Error processing dataset contents: {str(e)}")
        dataset.is_processed = False
        db.session.commit()
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
            try:
                # Get name and description from form
                name = request.form.get('name', '')
                if not name:
                    name = file.filename
                
                description = request.form.get('description', '')
                
                # Create unique filename to prevent overwriting
                original_filename = secure_filename(file.filename)
                unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{original_filename}"
                file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Save the file
                file.save(file_path)
                current_app.logger.info(f"File saved to {file_path}")
                
                # Create dataset record
                dataset = Dataset(
                    name=name,
                    description=description,
                    file_path=file_path,
                    file_type='xlsx',
                    user_id=current_user.id,
                    dataset_metadata={}  # Initialize with empty metadata
                )
                
                db.session.add(dataset)
                db.session.commit()
                current_app.logger.info(f"Dataset record created with ID {dataset.id}")
                
                # Process dataset asynchronously
                task = process_dataset.delay(dataset.id)
                
                # Store task ID in metadata
                dataset.dataset_metadata = {
                    'task_id': task.id,
                    'original_filename': original_filename,
                    'task_started_at': datetime.utcnow().isoformat()
                }
                db.session.commit()
                
                current_app.logger.info(f"Processing task {task.id} started for dataset {dataset.id}")
                flash('Dataset uploaded successfully! Processing has started.', 'success')
                return redirect(url_for('main.dataset_detail', dataset_id=dataset.id))
            except Exception as e:
                current_app.logger.error(f"Error uploading dataset: {str(e)}")
                flash(f'Error uploading dataset: {str(e)}', 'danger')
                return redirect(request.url)
        else:
            flash('File type not allowed. Please upload an Excel file (.xlsx).', 'danger')
            return redirect(request.url)
    
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
    
    try:
        # Always try to read metadata from the JSON field
        if dataset.dataset_metadata and 'columns' in dataset.dataset_metadata:
            metadata = dataset.dataset_metadata
        else:
            metadata = {'columns': []}
            
        # If dataset is processed, load preview data
        if dataset.is_processed:
            try:
                if os.path.exists(dataset.file_path):
                    df = pd.read_excel(dataset.file_path)
                    preview_data = df.head(10).to_dict('records')
                    columns = df.columns.tolist()
                else:
                    flash(f'Dataset file not found at {dataset.file_path}', 'danger')
                    dataset.is_processed = False
                    db.session.commit()
            except Exception as e:
                current_app.logger.error(f"Error loading dataset preview: {str(e)}")
                flash(f'Error loading dataset preview: {str(e)}', 'warning')
        
        # Trigger processing if not processed or processing failed
        if not dataset.is_processed:
            # Check if the file exists
            if os.path.exists(dataset.file_path):
                # Get the task ID if it exists in the metadata
                task_id = dataset.dataset_metadata.get('task_id') if dataset.dataset_metadata else None
                
                # Check if we already have a task in progress
                if not task_id:
                    # No task in progress, so start one
                    task = process_dataset.delay(dataset_id)
                    
                    # Update dataset metadata with task ID
                    if not dataset.dataset_metadata:
                        dataset.dataset_metadata = {}
                    
                    dataset.dataset_metadata['task_id'] = task.id
                    dataset.dataset_metadata['task_started_at'] = datetime.utcnow().isoformat()
                    db.session.commit()
                    
                    current_app.logger.info(f"Started processing task {task.id} for dataset {dataset_id}")
                    flash('Dataset processing has started. This page will automatically refresh when complete.', 'info')
            else:
                flash('Dataset file not found. Please upload again.', 'danger')
    except Exception as e:
        current_app.logger.error(f"Error in dataset_detail: {str(e)}")
        flash(f'An error occurred: {str(e)}', 'danger')
        metadata = {'columns': []}
    
    return render_template('datasets/detail.html', 
                           title=dataset.name,
                           dataset=dataset,
                           preview_data=preview_data,
                           columns=columns,
                           metadata=metadata)

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

@bp.route('/visualizations/<int:visualization_id>/export', methods=['GET'])
@login_required
def export_visualization(visualization_id):
    """Export visualization as PDF"""
    from flask import make_response
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    
    visualization = Visualization.query.get_or_404(visualization_id)

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
        # Make sure CSRF token is validated by Flask-WTF
        try:
            validate_csrf(request.form.get('csrf_token'))
        except Exception as e:
            current_app.logger.error(f"CSRF validation error: {str(e)}")
            flash('CSRF token validation failed. Please try again.', 'danger')
            return redirect(url_for('main.chat'))
            
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
    # Validate CSRF token when sent via AJAX
    if 'X-CSRFToken' in request.headers:
        from flask_wtf.csrf import validate_csrf
        try:
            validate_csrf(request.headers.get('X-CSRFToken'))
        except Exception as e:
            current_app.logger.error(f"CSRF validation error in chat: {str(e)}")
            return jsonify({'error': 'CSRF token validation failed'}), 400
    elif 'csrf_token' in request.form:
        from flask_wtf.csrf import validate_csrf
        try:
            validate_csrf(request.form.get('csrf_token'))
        except Exception as e:
            current_app.logger.error(f"CSRF validation error in chat form: {str(e)}")
            return jsonify({'error': 'CSRF token validation failed'}), 400
    
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
                try:
                    # Load dataset preview for context
                    df = pd.read_excel(dataset.file_path)
                    dataset_context = {
                        'name': dataset.name,
                        'rows': len(df),
                        'columns': df.columns.tolist(),
                        'preview': df.head(5).to_dict('records'),
                        'summary': df.describe().to_dict(),
                        'metadata': dataset.dataset_metadata
                    }
                except Exception as e:
                    current_app.logger.error(f"Error loading dataset for chat: {str(e)}")
                    dataset_context = None
        
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
        
        # Try to use OpenAI if configured
        ai_response = None
        model_used = 'rule-based'
        
        if current_app.config.get('OPENAI_API_KEY') and not current_app.config.get('OPENAI_API_KEY').startswith('sk-proj-'):
            try:
                import openai
                client = openai.Client(api_key=current_app.config.get('OPENAI_API_KEY'))
                
                # Prepare system prompt
                system_prompt = """You are an advanced AI assistant for data analysis and visualization. 
You communicate clearly and concisely with these guidelines:
1. Use markdown formatting for clear structure (headings, lists, etc.)
2. For code snippets, use markdown code blocks with appropriate language
3. For data insights, be specific and actionable
4. When suggesting visualizations, explain why they would be helpful
5. Keep responses concise but comprehensive"""
                
                if dataset_context:
                    cols_str = ', '.join(dataset_context['columns'])
                    system_prompt += f"\n\nYou have access to a dataset named '{dataset_context['name']}' with {dataset_context['rows']} rows and columns: {cols_str}."
                    
                    # Add data preview
                    system_prompt += "\n\nHere's a preview of the first few rows:"
                    preview_text = str(pd.DataFrame(dataset_context['preview']))
                    system_prompt += f"\n```\n{preview_text}\n```"
                    
                    # Add summary statistics if available
                    if dataset_context['summary']:
                        stats_text = ""
                        for col, stats in dataset_context['summary'].items():
                            if isinstance(stats, dict):  # Ensure it's a valid stats dictionary
                                stats_text += f"\n{col}:\n"
                                for stat, value in stats.items():
                                    stats_text += f"  {stat}: {value}\n"
                        if stats_text:
                            system_prompt += f"\n\nBasic statistics:\n```\n{stats_text}\n```"
                    
                    system_prompt += "\n\nYou can suggest Python code for analysis and visualizations based on this dataset."
                
                # Format messages for OpenAI
                openai_messages = [{"role": "system", "content": system_prompt}]
                
                # Add message history
                for msg in message_history:
                    openai_messages.append({"role": msg["role"], "content": msg["content"]})
                
                # Call OpenAI API
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=openai_messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                ai_response = response.choices[0].message.content
                model_used = "openai-gpt-3.5-turbo"
            except Exception as e:
                current_app.logger.error(f"OpenAI error: {str(e)}")
                # Fall back to rule-based response if OpenAI fails
        
        # If OpenAI failed or isn't configured, use our enhanced rule-based approach
        if ai_response is None:
            # Create a more conversational rule-based response system
            message_lower = message_content.lower()
            
            # Try to extract entities from the message
            data_terms = ['dataset', 'data', 'rows', 'columns', 'values', 'stats', 'statistics', 'information', 'details']
            viz_terms = ['visualize', 'visualization', 'chart', 'graph', 'plot', 'dashboard', 'display', 'show']
            analysis_terms = ['analyze', 'analysis', 'insights', 'trends', 'patterns', 'regression', 'correlation', 'compare', 'summarize']
            greeting_terms = ['hello', 'hi', 'hey', 'greetings', 'howdy', 'good morning', 'good afternoon', 'good evening']
            code_terms = ['code', 'script', 'python', 'function', 'example', 'implementation', 'program', 'snippet']
            help_terms = ['help', 'how to', 'tutorial', 'guide', 'explain', 'instruction', 'documentation']
            
            # Determine the message type
            contains_data_terms = any(term in message_lower for term in data_terms)
            contains_viz_terms = any(term in message_lower for term in viz_terms)
            contains_analysis_terms = any(term in message_lower for term in analysis_terms)
            contains_greeting = any(term in message_lower for term in greeting_terms)
            contains_code_terms = any(term in message_lower for term in code_terms)
            contains_help_terms = any(term in message_lower for term in help_terms)
            
            is_question = '?' in message_content or any(w in message_lower for w in ['what', 'how', 'when', 'where', 'why', 'can', 'could', 'would', 'will', 'is', 'are'])
            
            # Generate appropriate response based on message type
            if contains_greeting:
                greetings = [
                    "👋 Hello! I'm your AI data assistant. How can I help with your data analysis needs today?",
                    "Hi there! I'm ready to help you explore and understand your data. What would you like to know?",
                    "Greetings! I'm your data analysis AI. How can I assist you today?"
                ]
                ai_response = random.choice(greetings)
                
            elif contains_data_terms and dataset_context:
                if is_question:
                    responses = [
                        f"## Dataset: {dataset_context['name']}\n\nThis dataset has **{dataset_context['rows']} rows** and **{len(dataset_context['columns'])} columns**.\n\nThe columns are:\n\n" + 
                        "\n".join([f"- `{col}`" for col in dataset_context['columns']]) + 
                        "\n\nWhat specific aspect would you like to explore?",
                        
                        f"I've analyzed your dataset **{dataset_context['name']}**. It contains {dataset_context['rows']} records with {len(dataset_context['columns'])} fields.\n\n" +
                        "Here's a quick summary of the columns:\n" +
                        "\n".join([f"- `{col}`" for col in dataset_context['columns']]) +
                        "\n\nWould you like to see basic statistics, explore specific columns, or create visualizations?",
                        
                        f"Your dataset **{dataset_context['name']}** has {dataset_context['rows']} entries. The columns include:\n\n" +
                        "\n".join([f"- `{col}`" for col in dataset_context['columns']]) +
                        "\n\nWhat would you like to know about this data?"
                    ]
                else:
                    responses = [
                        "I can provide data analysis, but I need access to your dataset first. Would you like to link one to this chat?",
                        "For meaningful analysis, please link this chat to one of your datasets so I can access the data.",
                        "I'd be happy to analyze your data, but first we need to connect this chat to a dataset."
                    ]
                    ai_response = random.choice(responses)
                
            elif contains_data_terms and not dataset_context:
                responses = [
                    "I notice you're asking about data, but there's no dataset linked to this chat. Would you like to start a new chat with a dataset?",
                    "To analyze data, I'll need access to a dataset. You can create a new chat and link it to one of your datasets.",
                    "I can help with data analysis, but first we need to link this chat to a dataset. Would you like to do that?"
                ]
                ai_response = random.choice(responses)
                
            elif contains_viz_terms:
                if dataset_context:
                    numeric_columns = [col for col in dataset_context['columns'] 
                                     if any(t in str(type(dataset_context['preview'][0].get(col, ''))) 
                                          for t in ['int', 'float'])]
                    categorical_columns = [col for col in dataset_context['columns'] 
                                         if col not in numeric_columns]
                    
                    viz_examples = []
                    
                    if numeric_columns and categorical_columns:
                        viz_examples.append(
                            f"### Bar Chart\nCompare `{random.choice(numeric_columns)}` across different `{random.choice(categorical_columns)}` categories."
                        )
                        
                        if len(numeric_columns) >= 2:
                            viz_examples.append(
                                f"### Scatter Plot\nExplore the relationship between `{numeric_columns[0]}` and `{numeric_columns[1]}`."
                            )
                        
                        viz_examples.append(
                            f"### Pie Chart\nShow the distribution of `{random.choice(categorical_columns)}`."
                        )
                    
                    elif len(numeric_columns) >= 2:
                        viz_examples.append(
                            f"### Scatter Plot\nExplore the relationship between `{numeric_columns[0]}` and `{numeric_columns[1]}`."
                        )
                        viz_examples.append(
                            f"### Line Chart\nTrack changes in `{random.choice(numeric_columns)}` over time or sequence."
                        )
                    
                    viz_text = "\n\n".join(random.sample(viz_examples, min(2, len(viz_examples)))) if viz_examples else ""
                    
                    responses = [
                        f"## Visualization Suggestions\n\nBased on your dataset **{dataset_context['name']}**, here are some visualizations you could create:\n\n" +
                        viz_text + "\n\nYou can use the Visualizations tab to create these charts.",
                        
                        f"I can suggest some visualizations for your **{dataset_context['name']}** dataset:\n\n" +
                        viz_text + "\n\nWould you like to try creating one of these?",
                        
                        f"For your dataset **{dataset_context['name']}**, I recommend these visualizations:\n\n" +
                        viz_text + "\n\nThe Visualizations section of this app has tools to create these charts."
                    ]
                else:
                    responses = [
                        "I can help with data visualization, but first we need to link a dataset to this chat.",
                        "To create visualizations, please link this chat to one of your datasets or visit the Visualizations section.",
                        "Visualizations can provide powerful insights, but I need access to your data first. Would you like to link a dataset?"
                    ]
                ai_response = random.choice(responses)
                
            elif contains_analysis_terms:
                if dataset_context:
                    numeric_columns = [col for col in dataset_context['columns'] 
                                     if any(t in str(type(dataset_context['preview'][0].get(col, ''))) 
                                          for t in ['int', 'float'])]
                    
                    analysis_examples = []
                    
                    if numeric_columns:
                        if len(numeric_columns) >= 2:
                            analysis_examples.append(
                                f"### Correlation Analysis\nYou could explore correlations between `{numeric_columns[0]}` and `{numeric_columns[1]}`."
                            )
                        analysis_examples.append(
                            f"### Statistical Summary\nCompute basic statistics (mean, median, min, max) for `{random.choice(numeric_columns)}`."
                        )
                    
                    analysis_text = "\n\n".join(analysis_examples) if analysis_examples else ""
                    
                    responses = [
                        f"## Analysis Options for {dataset_context['name']}\n\n" +
                        "Here are some analyses you could perform:\n\n" +
                        analysis_text + "\n\n" +
                        "What specific analysis would you like to focus on?",
                        
                        f"For your **{dataset_context['name']}** dataset, I can help with:\n\n" +
                        analysis_text + "\n\n" +
                        "Let me know what aspects you'd like to analyze.",
                        
                        f"I can perform various analyses on your **{dataset_context['name']}** dataset:\n\n" +
                        analysis_text + "\n\n" +
                        "What insights are you looking for?"
                    ]
                else:
                    responses = [
                        "I can provide data analysis, but I need access to your dataset first. Would you like to link one to this chat?",
                        "For meaningful analysis, please link this chat to one of your datasets so I can access the data.",
                        "I'd be happy to analyze your data, but first we need to connect this chat to a dataset."
                    ]
                ai_response = random.choice(responses)
                
            elif contains_code_terms:
                if dataset_context:
                    numeric_columns = [col for col in dataset_context['columns'] 
                                     if any(t in str(type(dataset_context['preview'][0].get(col, ''))) 
                                          for t in ['int', 'float'])]
                    
                    sample_code = ""
                    
                    if numeric_columns:
                        col = random.choice(numeric_columns)
                        sample_code = f"""
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_excel('{dataset_context['name']}.xlsx')

# Basic statistics
stats = df['{col}'].describe()
print(stats)

# Create a histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='{col}', kde=True)
plt.title('{col} Distribution')
plt.show()
```
"""
                    
                    responses = [
                        f"## Sample Python Code for {dataset_context['name']}\n\n" +
                        "Here's a Python code example to analyze your data:\n" +
                        sample_code + "\n" +
                        "You can adapt this code to your specific needs. Would you like more examples?",
                        
                        f"Here's some Python code to help you analyze the **{dataset_context['name']}** dataset:\n" +
                        sample_code + "\n" +
                        "Let me know if you need more specific examples.",
                        
                        f"For your **{dataset_context['name']}** dataset, try this Python code:\n" +
                        sample_code + "\n" +
                        "What other code examples would you like to see?"
                    ]
                else:
                    responses = [
                        "I can provide code examples for data analysis, but I'll need access to your dataset first. Would you like to link one to this chat?",
                        "To give you relevant code examples, please link this chat to one of your datasets.",
                        "I'd be happy to provide code snippets, but first we need to connect this chat to a dataset."
                    ]
                ai_response = random.choice(responses)
                
            elif contains_help_terms:
                responses = [
                    "## How to Get Started\n\n" +
                    "Here's how you can use this platform:\n\n" +
                    "1. **Upload datasets** in the Datasets section\n" +
                    "2. **Create visualizations** in the Visualizations section\n" +
                    "3. **Chat with AI** about your data in this chat interface\n\n" +
                    "What would you like help with specifically?",
                    
                    "To get the most out of NextView.AI, try these steps:\n\n" +
                    "1. Upload your data in Excel format\n" +
                    "2. Use the visualization tools to create charts\n" +
                    "3. Ask me specific questions about your data\n\n" +
                    "Is there a particular feature you'd like to learn more about?",
                    
                    "Here's a quick guide to using this platform:\n\n" +
                    "- **Datasets**: Upload and manage your data files\n" +
                    "- **Visualizations**: Create charts and graphs from your data\n" +
                    "- **AI Chat**: Ask questions and get insights about your data\n\n" +
                    "What would you like to explore first?"
                ]
                ai_response = random.choice(responses)
                
            else:
                # General fallback responses
                responses = [
                    "I'm here to help with your data analysis and visualization needs. You can ask me about your datasets, request visualizations, or get insights from your data.",
                    "How can I assist with your data today? I can help analyze datasets, suggest visualizations, or answer questions about your data.",
                    "I'm your AI data assistant. I can help you understand your data, create visualizations, and extract insights. What would you like to explore?"
                ]
                ai_response = random.choice(responses)
        
        # Save AI response
        assistant_message = ChatMessage(
            session_id=session.id,
            content=ai_response,
            role='assistant',
            message_metadata={
                'generated_at': datetime.utcnow().isoformat(),
                'model': model_used
            }
        )
        db.session.add(assistant_message)
        db.session.commit()
        
        # If no title yet, generate one based on first user message
        if not session.title and len(previous_messages) <= 1:
            # Generate a concise title from the first message
            title_length = min(50, len(message_content))
            session.title = message_content[:title_length] + ('...' if len(message_content) > title_length else '')
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

@bp.route('/chat/<session_id>/generate', methods=['POST'])
@login_required
def generate_from_nlp(session_id):
    """Generate a report or visualization from an NLP query."""
    # Get the chat session
    session = ChatSession.query.filter_by(
        session_id=session_id, user_id=current_user.id
    ).first_or_404()
    
    # Check if dataset is linked
    if not session.dataset_id:
        return jsonify({"error": "No dataset linked to this chat session"}), 400
        
    # Get the query from the request
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
        
    query = data['query']
    
    try:
        # Load the dataset
        dataset = Dataset.query.get(session.dataset_id)
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404
            
        dataset_path = os.path.join(current_app.config['UPLOAD_FOLDER'], dataset.filename)
        
        # Create NLP processor
        processor = NLPProcessor(dataset_path)
        
        # Process the query
        report_html, image_data = processor.process_query(query)
        
        # Save user message
        user_message = ChatMessage(
            session_id=session.id,
            content=query,
            role='user'
        )
        db.session.add(user_message)
        
        # Save assistant message with the report
        response_content = ""
        
        if report_html:
            response_content += report_html
            
        if image_data:
            response_content += f'<img src="data:image/png;base64,{image_data}" alt="Data Visualization">'
            
        if not response_content:
            response_content = "I couldn't generate a report from that query. Could you be more specific?"
            
        assistant_message = ChatMessage(
            session_id=session.id,
            content=response_content,
            role='assistant'
        )
        db.session.add(assistant_message)
        db.session.commit()
        
        return jsonify({"success": True})
        
    except Exception as e:
        current_app.logger.error(f"Error generating report: {str(e)}")
        return jsonify({"error": str(e)}), 500

@bp.route('/nlp_test')
@login_required
def nlp_test():
    """Test page for the NLP processor."""
    # Get user's datasets for the dropdown
    datasets = Dataset.query.filter_by(user_id=current_user.id).all()
    return render_template('nlp_test.html', title="NLP Processor Test", datasets=datasets)