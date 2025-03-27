import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import logging
from datetime import datetime
from flask import current_app
from app.models.dataset import Dataset
from app.nlp.processor import NLPProcessor

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Advanced Data Processor for dataset preprocessing, feature analysis, and model building.
    This class handles the heavy lifting of data processing before AI analysis.
    """
    
    def __init__(self, dataset_id):
        """
        Initialize the data processor with a dataset ID.
        
        Args:
            dataset_id: The ID of the Dataset to process
        """
        self.dataset_id = dataset_id
        self.dataset = None
        self.df = None
        self.metadata = {}
        self.feature_stats = {}
        self.correlations = {}
        self.outliers = {}
        self.models = {}
        
    def load_dataset(self):
        """Load the dataset from database and file"""
        try:
            from app import db
            self.dataset = Dataset.query.get(self.dataset_id)
            
            if not self.dataset:
                logger.error(f"Dataset with ID {self.dataset_id} not found")
                return False
                
            if not os.path.exists(self.dataset.file_path):
                logger.error(f"Dataset file not found at {self.dataset.file_path}")
                return False
                
            # Load the dataset file
            logger.info(f"Loading dataset from {self.dataset.file_path}")
            self.df = pd.read_excel(self.dataset.file_path)
            
            # Initialize metadata if not present
            if not self.dataset.dataset_metadata:
                self.dataset.dataset_metadata = {}
                
            # Store basic dataset info
            self.metadata = self.dataset.dataset_metadata
            return True
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return False
    
    def preprocess_data(self):
        """Perform initial data preprocessing"""
        try:
            if self.df is None:
                if not self.load_dataset():
                    return False
            
            logger.info(f"Preprocessing dataset {self.dataset_id}")
            
            # Handle missing values
            missing_values = self.df.isnull().sum().to_dict()
            total_rows = len(self.df)
            
            # Store missing value information
            self.metadata['missing_values'] = {
                column: {
                    'count': int(count),
                    'percentage': round(float(count) / total_rows * 100, 2) if total_rows > 0 else 0
                } for column, count in missing_values.items() if count > 0
            }
            
            # Identify data types for each column
            dtypes = {}
            for col in self.df.columns:
                dtype = str(self.df[col].dtype)
                if dtype.startswith('int') or dtype.startswith('float'):
                    dtypes[col] = 'numeric'
                elif dtype == 'bool':
                    dtypes[col] = 'boolean'
                elif dtype == 'datetime64[ns]' or self._is_datetime(col):
                    dtypes[col] = 'datetime'
                    # Convert to datetime if not already
                    if dtype != 'datetime64[ns]':
                        try:
                            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                        except:
                            pass
                else:
                    dtypes[col] = 'categorical'
            
            self.metadata['column_types'] = dtypes
            return True
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return False
    
    def _is_datetime(self, column):
        """Check if a column contains datetime values"""
        if self.df[column].dtype == 'object':
            # Sample non-null values
            sample = self.df[column].dropna().head(5).tolist()
            if not sample:
                return False
                
            # Try to parse as datetime
            try:
                success = 0
                for val in sample:
                    if pd.to_datetime(val, errors='coerce') is not pd.NaT:
                        success += 1
                return success / len(sample) > 0.8  # If more than 80% parse as dates
            except:
                return False
        return False
    
    def analyze_features(self):
        """Perform detailed feature analysis"""
        try:
            if 'column_types' not in self.metadata:
                if not self.preprocess_data():
                    return False
            
            logger.info(f"Analyzing features for dataset {self.dataset_id}")
            
            # Analyze numeric columns
            numeric_stats = {}
            for col, dtype in self.metadata['column_types'].items():
                if dtype == 'numeric':
                    # Calculate statistics
                    stats = self.df[col].describe().to_dict()
                    # Add additional statistics
                    stats['skewness'] = float(self.df[col].skew())
                    stats['kurtosis'] = float(self.df[col].kurt())
                    
                    # Detect outliers using IQR method
                    Q1 = stats['25%']
                    Q3 = stats['75%']
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)][col]
                    
                    stats['outliers'] = {
                        'count': len(outliers),
                        'percentage': round(len(outliers) / len(self.df) * 100, 2),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound)
                    }
                    
                    numeric_stats[col] = stats
            
            # Analyze categorical columns
            categorical_stats = {}
            for col, dtype in self.metadata['column_types'].items():
                if dtype == 'categorical':
                    # Calculate value counts
                    value_counts = self.df[col].value_counts().head(10).to_dict()
                    # Convert keys to strings for JSON serialization
                    value_counts = {str(k): int(v) for k, v in value_counts.items()}
                    
                    categorical_stats[col] = {
                        'unique_values': int(self.df[col].nunique()),
                        'top_values': value_counts,
                        'mode': str(self.df[col].mode().iloc[0]) if not self.df[col].mode().empty else None
                    }
            
            # Analyze datetime columns
            datetime_stats = {}
            for col, dtype in self.metadata['column_types'].items():
                if dtype == 'datetime':
                    # Ensure column is datetime type
                    if self.df[col].dtype != 'datetime64[ns]':
                        try:
                            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                        except:
                            continue
                    
                    # Calculate statistics
                    datetime_stats[col] = {
                        'min_date': self.df[col].min().isoformat() if not pd.isna(self.df[col].min()) else None,
                        'max_date': self.df[col].max().isoformat() if not pd.isna(self.df[col].max()) else None,
                        'range_days': (self.df[col].max() - self.df[col].min()).days 
                            if not pd.isna(self.df[col].min()) and not pd.isna(self.df[col].max()) else None
                    }
            
            # Store all feature statistics
            self.feature_stats = {
                'numeric': numeric_stats,
                'categorical': categorical_stats,
                'datetime': datetime_stats
            }
            
            # Update metadata
            self.metadata['feature_stats'] = self.feature_stats
            return True
        except Exception as e:
            logger.error(f"Error analyzing features: {str(e)}")
            return False
    
    def calculate_correlations(self):
        """Calculate correlations between numeric features"""
        try:
            if 'feature_stats' not in self.metadata:
                if not self.analyze_features():
                    return False
            
            logger.info(f"Calculating correlations for dataset {self.dataset_id}")
            
            # Get numeric columns
            numeric_cols = [col for col, dtype in self.metadata['column_types'].items() 
                           if dtype == 'numeric']
            
            if len(numeric_cols) > 1:
                # Calculate correlation matrix
                corr_matrix = self.df[numeric_cols].corr().round(3).to_dict()
                
                # Convert to a more usable format
                correlations = []
                for col1 in numeric_cols:
                    for col2 in numeric_cols:
                        if col1 != col2 and abs(corr_matrix[col1][col2]) > 0.5:  # Only strong correlations
                            correlations.append({
                                'feature1': col1,
                                'feature2': col2,
                                'correlation': round(corr_matrix[col1][col2], 3)
                            })
                
                # Sort by absolute correlation value
                correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
                
                # Store correlations
                self.correlations = correlations
                self.metadata['correlations'] = correlations
            else:
                self.correlations = []
                self.metadata['correlations'] = []
            
            return True
        except Exception as e:
            logger.error(f"Error calculating correlations: {str(e)}")
            return False
    
    def detect_outliers(self):
        """Detect outliers in numeric features"""
        try:
            if 'feature_stats' not in self.metadata:
                if not self.analyze_features():
                    return False
            
            logger.info(f"Detecting outliers for dataset {self.dataset_id}")
            
            # Get numeric columns
            numeric_cols = [col for col, dtype in self.metadata['column_types'].items() 
                           if dtype == 'numeric']
            
            # Collect outlier information
            outliers = {}
            for col in numeric_cols:
                stats = self.metadata['feature_stats']['numeric'].get(col, {})
                if 'outliers' in stats and stats['outliers']['count'] > 0:
                    outliers[col] = stats['outliers']
            
            # Store outliers
            self.outliers = outliers
            self.metadata['outliers'] = outliers
            
            return True
        except Exception as e:
            logger.error(f"Error detecting outliers: {str(e)}")
            return False
    
    def generate_insights(self):
        """Generate automated insights from the data"""
        try:
            # Ensure all analyses are complete
            if 'outliers' not in self.metadata:
                if not self.detect_outliers():
                    return False
            
            logger.info(f"Generating insights for dataset {self.dataset_id}")
            
            insights = []
            
            # Insight 1: Data quality issues
            missing_data_cols = [col for col, info in self.metadata.get('missing_values', {}).items() 
                               if info['percentage'] > 5]
            if missing_data_cols:
                insights.append({
                    'type': 'data_quality',
                    'title': 'Missing Data Issues',
                    'description': f"Found significant missing data in {len(missing_data_cols)} columns. "
                                  f"Consider imputation or handling these missing values.",
                    'affected_columns': missing_data_cols
                })
            
            # Insight 2: Strong correlations
            strong_correlations = [c for c in self.metadata.get('correlations', []) 
                                 if abs(c['correlation']) > 0.7]
            if strong_correlations:
                insights.append({
                    'type': 'correlation',
                    'title': 'Strong Feature Correlations',
                    'description': f"Found {len(strong_correlations)} strong correlations between features. "
                                  f"This may indicate redundant information or interesting relationships.",
                    'correlations': strong_correlations[:5]  # Top 5 strongest
                })
            
            # Insight 3: Outliers
            outlier_cols = list(self.metadata.get('outliers', {}).keys())
            if outlier_cols:
                insights.append({
                    'type': 'outliers',
                    'title': 'Outliers Detected',
                    'description': f"Found outliers in {len(outlier_cols)} numeric columns. "
                                  f"These may represent errors or important edge cases.",
                    'affected_columns': outlier_cols
                })
            
            # Insight 4: Skewed distributions
            skewed_cols = [col for col, stats in self.metadata.get('feature_stats', {}).get('numeric', {}).items() 
                          if abs(stats.get('skewness', 0)) > 1]
            if skewed_cols:
                insights.append({
                    'type': 'distribution',
                    'title': 'Skewed Distributions',
                    'description': f"Found {len(skewed_cols)} columns with skewed distributions. "
                                  f"Consider transformations for these features.",
                    'affected_columns': skewed_cols
                })
            
            # Store insights
            self.metadata['insights'] = insights
            
            return True
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return False
    
    def save_results(self):
        """Save all processing results to the database"""
        try:
            from app import db
            
            # Add processing timestamp
            self.metadata['processed_at'] = datetime.utcnow().isoformat()
            self.metadata['processing_version'] = '1.0'
            
            # Update dataset metadata
            self.dataset.dataset_metadata = self.metadata
            self.dataset.is_processed = True
            
            # Save to database
            db.session.commit()
            logger.info(f"Successfully saved processing results for dataset {self.dataset_id}")
            
            return True
        except Exception as e:
            logger.error(f"Error saving processing results: {str(e)}")
            return False
    
    def process(self):
        """Run the complete data processing pipeline"""
        try:
            logger.info(f"Starting data processing pipeline for dataset {self.dataset_id}")
            
            # Step 1: Load dataset
            if not self.load_dataset():
                return False
                
            # Step 2: Preprocess data
            if not self.preprocess_data():
                return False
                
            # Step 3: Analyze features
            if not self.analyze_features():
                return False
                
            # Step 4: Calculate correlations
            if not self.calculate_correlations():
                return False
                
            # Step 5: Detect outliers
            if not self.detect_outliers():
                return False
                
            # Step 6: Generate insights
            if not self.generate_insights():
                return False
                
            # Step 7: Save results
            if not self.save_results():
                return False
                
            logger.info(f"Completed data processing pipeline for dataset {self.dataset_id}")
            return True
        except Exception as e:
            logger.error(f"Error in data processing pipeline: {str(e)}")
            return False