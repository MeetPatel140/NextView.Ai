from flask import jsonify, request, current_app, send_file
from flask_login import login_required, current_user
from app.api import bp
from app.models.dataset import Dataset, Visualization
from app import db, csrf
import pandas as pd
import os
import io
import openai
import json
from datetime import datetime
import matplotlib.pyplot as plt
import uuid
import base64
import logging
import traceback

# Exempt API routes from CSRF protection when using JSON but apply it for non-GET browser-based requests
@bp.before_request
def check_csrf():
    # Only apply CSRF protection to non-GET, non-HEAD, non-OPTIONS, non-TRACE methods
    if request.method not in ('GET', 'HEAD', 'OPTIONS', 'TRACE'):
        # Check if this is a request with a custom header (API client)
        if not request.headers.get('X-CSRFToken'):
            # No CSRF token in headers - API should use it for browser-based requests
            if request.content_type and 'application/json' not in request.content_type:
                # Not JSON API, validate CSRF
                csrf.protect()

@bp.route('/datasets', methods=['GET'])
@login_required
def get_datasets():
    """API endpoint to get all datasets for the current user"""
    datasets = Dataset.query.filter_by(user_id=current_user.id).all()
    return jsonify({
        'datasets': [{
            'id': dataset.id,
            'name': dataset.name,
            'description': dataset.description,
            'created_at': dataset.created_at.isoformat(),
            'row_count': dataset.row_count,
            'column_count': dataset.column_count
        } for dataset in datasets]
    })

@bp.route('/datasets/<int:dataset_id>', methods=['GET'])
@login_required
def get_dataset(dataset_id):
    """API endpoint to get a specific dataset"""
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()
    
    # Read the dataset file to get a preview
    try:
        df = pd.read_excel(dataset.file_path)
        preview = df.head(5).to_dict(orient='records')
        columns = df.columns.tolist()
    except Exception as e:
        preview = []
        columns = []
    
    return jsonify({
        'dataset': {
            'id': dataset.id,
            'name': dataset.name,
            'description': dataset.description,
            'created_at': dataset.created_at.isoformat(),
            'row_count': dataset.row_count,
            'column_count': dataset.column_count,
            'columns': columns,
            'preview': preview
        }
    })

@bp.route('/datasets/<int:dataset_id>/status', methods=['GET'])
@login_required
def get_dataset_status(dataset_id):
    """API endpoint to check the processing status of a dataset"""
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()
    
    # Check file existence
    file_exists = os.path.exists(dataset.file_path) if dataset.file_path else False
    
    # Process metadata
    metadata = {}
    task_id = None
    task_state = 'UNKNOWN'
    
    if dataset.dataset_metadata and isinstance(dataset.dataset_metadata, dict):
        metadata = dataset.dataset_metadata
        task_id = metadata.get('task_id')
    
    # If file is missing but dataset is marked as processed, update status
    if not file_exists and dataset.is_processed:
        dataset.is_processed = False
        db.session.commit()
    
    # If the dataset has a task ID, check its status
    if task_id and not dataset.is_processed:
        try:
            from app import celery
            task_result = celery.AsyncResult(task_id)
            task_state = task_result.state
            
            # If task is done, but dataset not marked as processed, something went wrong
            if task_state == 'SUCCESS' and not dataset.is_processed:
                # Import the task and try processing again
                from app.main.routes import process_dataset
                
                # Only process again if it's been more than 60 seconds since last attempt
                should_retry = True
                if 'task_started_at' in metadata:
                    try:
                        from datetime import datetime, timedelta
                        task_started_at = datetime.fromisoformat(metadata['task_started_at'])
                        time_since_start = datetime.utcnow() - task_started_at
                        should_retry = time_since_start > timedelta(seconds=60)
                    except:
                        pass
                
                if should_retry:
                    # Start a new task
                    task = process_dataset.delay(dataset_id)
                    metadata['task_id'] = task.id
                    metadata['task_started_at'] = datetime.utcnow().isoformat()
                    dataset.dataset_metadata = metadata
                    db.session.commit()
                    current_app.logger.info(f"Retrying processing for dataset {dataset_id} with task {task.id}")
            
            # If task failed, and it's been a while, retry
            elif task_state in ('FAILURE', 'REVOKED'):
                # Only retry if we have last_triggered parameter and it's been a while
                if 'last_triggered' in request.args:
                    try:
                        last_triggered = float(request.args.get('last_triggered', 0))
                        now = datetime.utcnow().timestamp()
                        
                        # Only retry if more than 60 seconds since last trigger
                        if (now - last_triggered) > 60:
                            from app.main.routes import process_dataset
                            task = process_dataset.delay(dataset_id)
                            metadata['task_id'] = task.id
                            metadata['task_started_at'] = datetime.utcnow().isoformat()
                            dataset.dataset_metadata = metadata
                            db.session.commit()
                            current_app.logger.info(f"Retrying processing for dataset {dataset_id} with task {task.id}")
                    except Exception as e:
                        current_app.logger.error(f"Error checking task status: {str(e)}")
        except Exception as e:
            current_app.logger.error(f"Error checking Celery task: {str(e)}")
    
    # If dataset is not processed and the file exists but no task ID, start processing
    elif not dataset.is_processed and file_exists and not task_id:
        # Only trigger if it hasn't been triggered recently
        should_process = True
        if 'last_triggered' in request.args:
            try:
                last_triggered = float(request.args.get('last_triggered', 0))
                now = datetime.utcnow().timestamp()
                
                # Only trigger if more than 30 seconds since last trigger
                should_process = (now - last_triggered) > 30
            except (ValueError, TypeError):
                pass
        
        if should_process:
            from app.main.routes import process_dataset
            task = process_dataset.delay(dataset_id)
            
            # Update metadata with task info
            if not metadata:
                metadata = {}
            
            metadata['task_id'] = task.id
            metadata['task_started_at'] = datetime.utcnow().isoformat()
            dataset.dataset_metadata = metadata
            db.session.commit()
            current_app.logger.info(f"Started processing for dataset {dataset_id} with task {task.id}")
    
    # Return detailed status information
    return jsonify({
        'id': dataset.id,
        'name': dataset.name,
        'is_processed': dataset.is_processed,
        'file_exists': file_exists,
        'row_count': dataset.row_count,
        'column_count': dataset.column_count,
        'task_id': task_id,
        'task_state': task_state,
        'has_metadata': bool(metadata),
        'timestamp': datetime.utcnow().timestamp()
    })

@bp.route('/datasets/<int:dataset_id>/download', methods=['GET'])
@login_required
def download_dataset(dataset_id):
    """API endpoint to download a dataset in CSV or Excel format"""
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()
    
    if not dataset.is_processed:
        return jsonify({
            'error': 'Dataset is still processing'
        }), 400
    
    try:
        df = pd.read_excel(dataset.file_path)
        
        # Determine format (default to Excel)
        format_type = request.args.get('format', 'excel').lower()
        
        if format_type == 'csv':
            # Create a buffer for the CSV data
            buffer = io.StringIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)
            
            # Create response with CSV data
            csv_data = buffer.getvalue()
            mem_buffer = io.BytesIO()
            mem_buffer.write(csv_data.encode())
            mem_buffer.seek(0)
            
            return send_file(
                mem_buffer,
                mimetype='text/csv',
                as_attachment=True,
                download_name=f"{dataset.name}.csv"
            )
        else:
            # Create a buffer for the Excel data
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False)
            buffer.seek(0)
            
            return send_file(
                buffer,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=f"{dataset.name}.xlsx"
            )
    except Exception as e:
        current_app.logger.error(f"Error downloading dataset: {str(e)}")
        return jsonify({
            'error': f'Error downloading dataset: {str(e)}'
        }), 500

@bp.route('/datasets/<int:dataset_id>/ai-query', methods=['POST'])
@login_required
def dataset_ai_query(dataset_id):
    """API endpoint to query a dataset using AI"""
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()
    
    if not dataset.is_processed:
        return jsonify({
            'error': 'Dataset is still processing'
        }), 400
    
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'Query is required'}), 400
    
    try:
        # Load dataset into pandas
        df = pd.read_excel(dataset.file_path)
        
        # Get dataset statistics and sample data
        desc = df.describe().to_string()
        sample = df.head(5).to_string()
        
        # Prepare the query with context
        query = data['query']
        
        # Check if OpenAI API key is available and valid
        openai_api_key = current_app.config.get('OPENAI_API_KEY')
        if openai_api_key and not openai_api_key.startswith('sk-proj-'):
            # Initialize OpenAI client
            client = openai.Client(api_key=openai_api_key)
            
            # Context for API call
            context = f"""
Dataset Name: {dataset.name}
Description: {dataset.description}
Shape: {df.shape[0]} rows x {df.shape[1]} columns
Columns: {', '.join(df.columns.tolist())}
Statistics:
{desc}

Sample Data:
{sample}

Question: {query}
"""
            
            # Get AI response
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data analysis assistant. Provide concise insights about the dataset based on the question."},
                    {"role": "user", "content": context}
                ]
            )
            
            # Format response with markdown
            ai_response = response.choices[0].message.content
        else:
            # Generate a fallback response without OpenAI
            columns_str = ', '.join(df.columns.tolist())
            
            # Generate simple rule-based response
            if 'trend' in query.lower() or 'pattern' in query.lower():
                ai_response = f"Based on your dataset '{dataset.name}' with {df.shape[0]} rows and {df.shape[1]} columns, I can see some interesting patterns. The data includes columns: {columns_str}. To fully analyze trends, I would need to perform specific statistical tests on your dataset."
            elif 'summary' in query.lower() or 'describe' in query.lower():
                ai_response = f"Here's a summary of your dataset '{dataset.name}':\n\n- Rows: {df.shape[0]}\n- Columns: {df.shape[1]}\n- Column names: {columns_str}\n\nFor more detailed analysis, you could explore the statistics shown in the Dataset Information panel."
            elif 'column' in query.lower() or 'field' in query.lower():
                ai_response = f"Your dataset contains the following columns: {columns_str}. Each column has specific data characteristics that can be analyzed further."
            elif 'missing' in query.lower() or 'null' in query.lower():
                missing_counts = df.isna().sum().to_dict()
                missing_info = '\n'.join([f"- {col}: {count} missing values" for col, count in missing_counts.items() if count > 0])
                if missing_info:
                    ai_response = f"I found the following missing values in your dataset:\n\n{missing_info}"
                else:
                    ai_response = "Great news! Your dataset doesn't have any missing values."
            else:
                ai_response = f"I analyzed the dataset '{dataset.name}' with {df.shape[0]} rows and {df.shape[1]} columns. The columns include {columns_str}. To provide more specific insights, please ask about particular aspects of your data such as trends, summaries, or specific columns."
        
        return jsonify({
            'response': ai_response
        })
    except Exception as e:
        current_app.logger.error(f"Error in AI query: {str(e)}")
        return jsonify({
            'error': f'Error processing AI query: {str(e)}',
            'response': 'Sorry, I encountered an error analyzing your dataset.'
        }), 500

@bp.route('/visualizations', methods=['GET'])
@login_required
def get_visualizations():
    """API endpoint to get all visualizations for the current user"""
    visualizations = Visualization.query.filter_by(user_id=current_user.id).all()
    return jsonify({
        'visualizations': [{
            'id': viz.id,
            'name': viz.name,
            'description': viz.description,
            'created_at': viz.created_at.isoformat(),
            'chart_type': viz.chart_type,
            'dataset_id': viz.dataset_id
        } for viz in visualizations]
    })

@bp.route('/datasets/<int:dataset_id>', methods=['DELETE'])
@login_required
def delete_dataset(dataset_id):
    """API endpoint to delete a dataset"""
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()
    
    try:
        # Delete the file if it exists
        if dataset.file_path and os.path.exists(dataset.file_path):
            try:
                os.remove(dataset.file_path)
                current_app.logger.info(f"Deleted file {dataset.file_path}")
            except Exception as e:
                current_app.logger.error(f"Error deleting file {dataset.file_path}: {str(e)}")
        
        # Get dataset name for the success message
        dataset_name = dataset.name
        
        # Delete dataset from DB (this will cascade delete visualizations)
        db.session.delete(dataset)
        db.session.commit()
        
        current_app.logger.info(f"Dataset {dataset_id} deleted successfully")
        
        return jsonify({
            'success': True,
            'message': f"Dataset '{dataset_name}' deleted successfully"
        })
    except Exception as e:
        current_app.logger.error(f"Error deleting dataset {dataset_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Error deleting dataset: {str(e)}"
        }), 500

@bp.route('/test_nlp', methods=['POST'])
@login_required
def test_nlp():
    """
    Test route for the NLP processor.
    """
    try:
        # Get data from request
        data = request.get_json()
        
        if not data or 'dataset_id' not in data or 'query' not in data:
            return jsonify({"error": "Missing dataset_id or query"}), 400
            
        dataset_id = data['dataset_id']
        query = data['query']
        
        # Get the dataset
        dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first()
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404
            
        dataset_path = os.path.join(current_app.config['UPLOAD_FOLDER'], dataset.filename)
        
        # Create NLP processor
        from app.nlp.processor import NLPProcessor
        processor = NLPProcessor(dataset_path)
        
        # Process the query
        report_html, image_data = processor.process_query(query)
        
        response = {
            "success": True,
            "report_html": report_html,
            "image_data": image_data
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        current_app.logger.error(f"Error testing NLP processor: {str(e)}")
        return jsonify({"error": str(e)}), 500