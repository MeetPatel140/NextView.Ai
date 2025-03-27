from flask import jsonify, request, current_app
from flask_login import login_required, current_user
from app.api import bp
from app.models.dataset import Dataset, Visualization
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