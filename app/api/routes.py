from flask import jsonify, request, current_app
from flask import jsonify, request, current_app
from flask_login import login_required, current_user
from app.api import bp
from app.models.dataset import Dataset, Visualization
from app import db
from app import db
import pandas as pd
import os

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

@bp.route('/datasets/<int:dataset_id>', methods=['DELETE'])
@login_required
def delete_dataset(dataset_id):
    """API endpoint to delete a specific dataset"""
    dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()
    
    try:
        # Delete the dataset file if it exists
        if os.path.exists(dataset.file_path):
            os.remove(dataset.file_path)
        
        # Delete the dataset from database
        db.session.delete(dataset)
        db.session.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

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