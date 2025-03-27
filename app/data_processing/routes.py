from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from app import db
from app.models.dataset import Dataset
from app.data_processing.tasks import process_dataset_advanced, reprocess_dataset
from datetime import datetime
import logging

bp = Blueprint('data_processing', __name__)
logger = logging.getLogger(__name__)

@bp.route('/process/<int:dataset_id>', methods=['POST'])
@login_required
def process_dataset(dataset_id):
    """
    Start advanced processing for a dataset
    """
    try:
        # Get dataset
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # Check if user owns the dataset
        if dataset.user_id != current_user.id:
            return jsonify({'error': 'You do not have permission to process this dataset'}), 403
        
        # Check if file exists
        import os
        if not os.path.exists(dataset.file_path):
            return jsonify({'error': 'Dataset file not found'}), 404
        
        # Start processing task
        task = process_dataset_advanced.delay(dataset_id)
        
        # Update dataset metadata
        if not dataset.dataset_metadata:
            dataset.dataset_metadata = {}
            
        dataset.dataset_metadata['processing_status'] = 'queued'
        dataset.dataset_metadata['processing_task_id'] = task.id
        dataset.dataset_metadata['processing_queued_at'] = datetime.utcnow().isoformat()
        db.session.commit()
        
        logger.info(f"Advanced processing queued for dataset {dataset_id} with task {task.id}")
        
        return jsonify({
            'message': 'Dataset processing started',
            'task_id': task.id,
            'status': 'queued'
        }), 202
    except Exception as e:
        logger.error(f"Error starting dataset processing: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/reprocess/<int:dataset_id>', methods=['POST'])
@login_required
def reprocess_dataset_route(dataset_id):
    """
    Reprocess a dataset that has already been processed
    """
    try:
        # Get dataset
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # Check if user owns the dataset
        if dataset.user_id != current_user.id:
            return jsonify({'error': 'You do not have permission to reprocess this dataset'}), 403
        
        # Check if file exists
        import os
        if not os.path.exists(dataset.file_path):
            return jsonify({'error': 'Dataset file not found'}), 404
        
        # Start reprocessing task
        task = reprocess_dataset.delay(dataset_id)
        
        # Update dataset metadata
        if not dataset.dataset_metadata:
            dataset.dataset_metadata = {}
            
        dataset.dataset_metadata['processing_status'] = 'queued'
        dataset.dataset_metadata['processing_task_id'] = task.id
        dataset.dataset_metadata['processing_queued_at'] = datetime.utcnow().isoformat()
        db.session.commit()
        
        logger.info(f"Reprocessing queued for dataset {dataset_id} with task {task.id}")
        
        return jsonify({
            'message': 'Dataset reprocessing started',
            'task_id': task.id,
            'status': 'queued'
        }), 202
    except Exception as e:
        logger.error(f"Error starting dataset reprocessing: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/status/<int:dataset_id>', methods=['GET'])
@login_required
def get_processing_status(dataset_id):
    """
    Get the current processing status of a dataset
    """
    try:
        # Get dataset
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # Check if user owns the dataset
        if dataset.user_id != current_user.id:
            return jsonify({'error': 'You do not have permission to view this dataset'}), 403
        
        # Get processing status from metadata
        metadata = dataset.dataset_metadata or {}
        processing_status = metadata.get('processing_status', 'unknown')
        processing_task_id = metadata.get('processing_task_id')
        
        # If there's a task ID, check its status
        task_status = None
        if processing_task_id:
            from app import celery
            task_result = celery.AsyncResult(processing_task_id)
            task_status = task_result.state
        
        # Compile status information
        status_info = {
            'dataset_id': dataset_id,
            'is_processed': dataset.is_processed,
            'processing_status': processing_status,
            'task_id': processing_task_id,
            'task_status': task_status
        }
        
        # Add timestamps if available
        for key in ['processing_queued_at', 'processing_started_at', 'processing_completed_at']:
            if key in metadata:
                status_info[key] = metadata[key]
        
        # Add error information if available
        if 'processing_error' in metadata:
            status_info['error'] = metadata['processing_error']
        
        return jsonify(status_info), 200
    except Exception as e:
        logger.error(f"Error getting processing status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/insights/<int:dataset_id>', methods=['GET'])
@login_required
def get_dataset_insights(dataset_id):
    """
    Get the insights generated during dataset processing
    """
    try:
        # Get dataset
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # Check if user owns the dataset
        if dataset.user_id != current_user.id:
            return jsonify({'error': 'You do not have permission to view this dataset'}), 403
        
        # Check if dataset is processed
        if not dataset.is_processed:
            return jsonify({'error': 'Dataset has not been processed yet'}), 400
        
        # Get insights from metadata
        metadata = dataset.dataset_metadata or {}
        insights = metadata.get('insights', [])
        
        return jsonify({
            'dataset_id': dataset_id,
            'insights': insights
        }), 200
    except Exception as e:
        logger.error(f"Error getting dataset insights: {str(e)}")
        return jsonify({'error': str(e)}), 500