from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from app.models.dataset import Dataset
from app.nlp.advanced_processor import AdvancedNLPProcessor
import logging

bp = Blueprint('data_visualization', __name__)
logger = logging.getLogger(__name__)

@bp.route('/visualize', methods=['POST'])
@login_required
def visualize_data():
    """Generate visualization from natural language query"""
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    
    query = data['query']
    dataset_id = data.get('dataset_id')
    
    try:
        # Find the dataset to query
        if dataset_id:
            dataset = Dataset.query.get(dataset_id)
            if not dataset or dataset.user_id != current_user.id:
                return jsonify({'error': 'Dataset not found or access denied'}), 404
        else:
            # Use the most recent dataset owned by the user
            dataset = Dataset.query.filter_by(user_id=current_user.id).order_by(Dataset.updated_at.desc()).first()
            if not dataset:
                return jsonify({'error': 'No datasets found. Please upload a dataset first.'}), 404
        
        # Process the query
        processor = AdvancedNLPProcessor(dataset)
        result = processor.process_query(query)
        
        # Create a new visualization record if successful
        if result.get('visualization'):
            from app.models.dataset import Visualization
            from datetime import datetime
            
            # Determine chart type from metadata
            chart_type = result.get('metadata', {}).get('viz_type', 'bar')
            
            # Create visualization config
            config = {
                'query': query,
                'chart_type': chart_type,
                'insights': result.get('insights', []),
                'processing_method': result.get('metadata', {}).get('processing_method', 'rule_based')
            }
            
            # Create and save visualization
            viz = Visualization(
                name=f"Visualization of {dataset.name} - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                description=query,
                chart_type=chart_type,
                config=config,
                dataset_id=dataset.id,
                user_id=current_user.id
            )
            
            from app import db
            db.session.add(viz)
            db.session.commit()
            
            # Add visualization ID to result
            result['visualization_id'] = viz.id
        
        return jsonify({
            'success': True,
            'html': result['html'],
            'visualization': result['visualization'],
            'insights': result.get('insights', []),
            'dataset_name': dataset.name,
            'dataset_id': dataset.id,
            'metadata': result.get('metadata', {})
        })
        
    except Exception as e:
        logger.error(f"Error in visualization API: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bp.route('/datasets', methods=['GET'])
@login_required
def get_user_datasets():
    """Get all datasets for the current user"""
    datasets = Dataset.query.filter_by(user_id=current_user.id).order_by(Dataset.updated_at.desc()).all()
    
    return jsonify({
        'datasets': [{
            'id': dataset.id,
            'name': dataset.name,
            'description': dataset.description,
            'created_at': dataset.created_at.isoformat(),
            'updated_at': dataset.updated_at.isoformat(),
            'row_count': dataset.row_count,
            'column_count': dataset.column_count
        } for dataset in datasets]
    })