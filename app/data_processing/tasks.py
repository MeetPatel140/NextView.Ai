import logging
from app import celery, db
from app.data_processing.processor import DataProcessor
from app.models.dataset import Dataset
from datetime import datetime

logger = logging.getLogger(__name__)

@celery.task(name='app.data_processing.tasks.process_dataset_advanced', bind=True)
def process_dataset_advanced(self, dataset_id):
    """
    Advanced dataset processing task that performs comprehensive analysis.
    This task handles preprocessing, feature analysis, and insight generation.
    
    Args:
        dataset_id: The ID of the dataset to process
    """
    try:
        logger.info(f"Starting advanced processing for dataset {dataset_id}")
        
        # Get dataset
        dataset = Dataset.query.get(dataset_id)
        if not dataset:
            logger.error(f"Dataset with ID {dataset_id} not found")
            return False
        
        # Update task status in metadata
        if not dataset.dataset_metadata:
            dataset.dataset_metadata = {}
            
        dataset.dataset_metadata['processing_status'] = 'running'
        dataset.dataset_metadata['processing_started_at'] = datetime.utcnow().isoformat()
        dataset.dataset_metadata['processing_task_id'] = self.request.id
        db.session.commit()
        
        # Create processor and run pipeline
        processor = DataProcessor(dataset_id)
        success = processor.process()
        
        # Update processing status
        if success:
            dataset.dataset_metadata['processing_status'] = 'completed'
            dataset.dataset_metadata['processing_completed_at'] = datetime.utcnow().isoformat()
            dataset.is_processed = True
        else:
            dataset.dataset_metadata['processing_status'] = 'failed'
            dataset.dataset_metadata['processing_error'] = 'Failed to process dataset'
            dataset.is_processed = False
            
        db.session.commit()
        logger.info(f"Advanced processing for dataset {dataset_id} {'completed' if success else 'failed'}")
        return success
    except Exception as e:
        logger.exception(f"Error in advanced processing for dataset {dataset_id}: {str(e)}")
        
        # Update error status
        try:
            dataset = Dataset.query.get(dataset_id)
            if dataset and dataset.dataset_metadata:
                dataset.dataset_metadata['processing_status'] = 'failed'
                dataset.dataset_metadata['processing_error'] = str(e)
                dataset.is_processed = False
                db.session.commit()
        except Exception as inner_e:
            logger.error(f"Failed to update error status: {str(inner_e)}")
            
        return False

@celery.task(name='app.data_processing.tasks.reprocess_dataset', bind=True)
def reprocess_dataset(self, dataset_id):
    """
    Reprocess a dataset that has already been processed before.
    This is useful when the processing pipeline has been updated or when
    the dataset has been modified.
    
    Args:
        dataset_id: The ID of the dataset to reprocess
    """
    try:
        logger.info(f"Reprocessing dataset {dataset_id}")
        
        # Get dataset
        dataset = Dataset.query.get(dataset_id)
        if not dataset:
            logger.error(f"Dataset with ID {dataset_id} not found")
            return False
        
        # Reset processing status
        if not dataset.dataset_metadata:
            dataset.dataset_metadata = {}
            
        # Preserve original metadata but update processing status
        dataset.dataset_metadata['previous_processing'] = {
            'status': dataset.dataset_metadata.get('processing_status'),
            'completed_at': dataset.dataset_metadata.get('processing_completed_at'),
            'task_id': dataset.dataset_metadata.get('processing_task_id')
        }
        
        dataset.dataset_metadata['processing_status'] = 'running'
        dataset.dataset_metadata['processing_started_at'] = datetime.utcnow().isoformat()
        dataset.dataset_metadata['processing_task_id'] = self.request.id
        dataset.is_processed = False
        db.session.commit()
        
        # Create processor and run pipeline
        processor = DataProcessor(dataset_id)
        success = processor.process()
        
        # Update processing status
        if success:
            dataset.dataset_metadata['processing_status'] = 'completed'
            dataset.dataset_metadata['processing_completed_at'] = datetime.utcnow().isoformat()
            dataset.is_processed = True
        else:
            dataset.dataset_metadata['processing_status'] = 'failed'
            dataset.dataset_metadata['processing_error'] = 'Failed to reprocess dataset'
            dataset.is_processed = False
            
        db.session.commit()
        logger.info(f"Reprocessing for dataset {dataset_id} {'completed' if success else 'failed'}")
        return success
    except Exception as e:
        logger.exception(f"Error in reprocessing dataset {dataset_id}: {str(e)}")
        
        # Update error status
        try:
            dataset = Dataset.query.get(dataset_id)
            if dataset and dataset.dataset_metadata:
                dataset.dataset_metadata['processing_status'] = 'failed'
                dataset.dataset_metadata['processing_error'] = str(e)
                dataset.is_processed = False
                db.session.commit()
        except Exception as inner_e:
            logger.error(f"Failed to update error status: {str(inner_e)}")
            
        return False