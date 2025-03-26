from flask import current_app, jsonify
from app.models.dataset import Dataset
from app.nlp.advanced_processor import AdvancedNLPProcessor
from app.chatbot.visualization_generator import VisualizationGenerator
import logging

logger = logging.getLogger(__name__)

class ChatbotNLPIntegration:
    """
    Integration class to connect the chatbot with the advanced NLP processor
    for natural language-based data visualization and reporting.
    """
    
    @staticmethod
    def process_data_query(query, dataset_id=None):
        """
        Process a natural language query about data and generate visualizations/insights.
        
        Args:
            query (str): The natural language query from the user
            dataset_id (int, optional): The ID of the dataset to query. If None, will try to
                                        detect a dataset from the query or use the most recent.
        
        Returns:
            dict: A response dictionary containing:
                - success: Boolean indicating if processing was successful
                - html: HTML content to display in the chat
                - visualization: Base64 encoded image if a visualization was generated
                - insights: List of key insights extracted from the data
                - error: Error message if processing failed
        """
        try:
            # Find the dataset to query
            dataset = None
            
            if dataset_id:
                # If dataset_id is provided, use that dataset
                dataset = Dataset.query.get(dataset_id)
                if not dataset:
                    return {
                        'success': False,
                        'error': f"Dataset with ID {dataset_id} not found"
                    }
            else:
                # Try to find the most recently updated dataset
                dataset = Dataset.query.order_by(Dataset.updated_at.desc()).first()
                
                if not dataset:
                    return {
                        'success': False,
                        'error': "No datasets found. Please upload a dataset first."
                    }
            
            # Initialize the advanced NLP processor with the dataset
            processor = AdvancedNLPProcessor(dataset)
            
            # Process the query
            result = processor.process_query(query)
            
            # Return the response
            return {
                'success': True,
                'html': result['html'],
                'visualization': result['visualization'],
                'insights': result['insights'],
                'dataset_name': dataset.name,
                'metadata': result.get('metadata', {})
            }
            
        except Exception as e:
            logger.error(f"Error processing data query: {str(e)}")
            return {
                'success': False,
                'error': f"Error processing your query: {str(e)}"
            }
    
    @staticmethod
    def detect_data_query(message):
        """
        Detect if a message is a data query that should be processed by the NLP processor.
        
        Args:
            message (str): The message to analyze
            
        Returns:
            bool: True if the message appears to be a data query, False otherwise
        """
        # List of terms that indicate a data query
        data_query_terms = [
            'chart', 'graph', 'plot', 'visualization', 'visualize',
            'show me', 'display', 'analyze', 'analysis', 'trend',
            'compare', 'comparison', 'statistics', 'stats', 'data',
            'report', 'summary', 'insights', 'correlation', 'relationship',
            'distribution', 'average', 'mean', 'median', 'mode',
            'maximum', 'minimum', 'count', 'sum', 'total'
        ]
        
        # Check if any of the terms are in the message
        message_lower = message.lower()
        return any(term in message_lower for term in data_query_terms)
    
    @staticmethod
    def generate_comprehensive_report(query, dataset_id=None):
        """
        Generate a comprehensive report with multiple visualizations based on a natural language query.
        
        Args:
            query (str): The natural language query from the user
            dataset_id (int, optional): The ID of the dataset to query
            
        Returns:
            dict: A response dictionary containing report data
        """
        try:
            # Find the dataset to query
            dataset = None
            
            if dataset_id:
                dataset = Dataset.query.get(dataset_id)
                if not dataset:
                    return {
                        'success': False,
                        'error': f"Dataset with ID {dataset_id} not found"
                    }
            else:
                dataset = Dataset.query.order_by(Dataset.updated_at.desc()).first()
                
                if not dataset:
                    return {
                        'success': False,
                        'error': "No datasets found. Please upload a dataset first."
                    }
            
            # Initialize the visualization generator
            viz_generator = VisualizationGenerator(dataset)
            
            # Generate the comprehensive report
            result = viz_generator.generate_report(query)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {str(e)}")
            return {
                'success': False,
                'error': f"Error generating report: {str(e)}"
            }
    
    @staticmethod
    def detect_report_query(message):
        """
        Detect if a message is requesting a comprehensive report.
        
        Args:
            message (str): The message to analyze
            
        Returns:
            bool: True if the message appears to be requesting a report, False otherwise
        """
        # List of terms that indicate a report query
        report_query_terms = [
            'comprehensive report', 'detailed report', 'full report',
            'complete analysis', 'deep dive', 'in-depth analysis',
            'generate a report', 'create a report', 'make a report',
            'dashboard', 'overview', 'summary report'
        ]
        
        # Check if any of the terms are in the message
        message_lower = message.lower()
        return any(term in message_lower for term in report_query_terms)