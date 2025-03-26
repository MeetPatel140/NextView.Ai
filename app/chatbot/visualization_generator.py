import base64
import io
import logging
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from app.models.dataset import Dataset
from app.nlp.advanced_processor import AdvancedNLPProcessor

logger = logging.getLogger(__name__)

class VisualizationGenerator:
    """
    Class for generating visualizations from natural language queries.
    This class extends the capabilities of the ChatbotNLPIntegration by providing
    more specialized visualization and reporting features.
    """
    
    def __init__(self, dataset=None):
        """
        Initialize the visualization generator with an optional dataset.
        
        Args:
            dataset: The Dataset model object to use for visualization
        """
        self.dataset = dataset
        self.df = None
        self.processor = None
        
        if dataset:
            self.load_dataset()
    
    def load_dataset(self):
        """
        Load the dataset and initialize the NLP processor
        """
        try:
            self.processor = AdvancedNLPProcessor(self.dataset)
            self.df = self.processor.df
            logger.info(f"Loaded dataset {self.dataset.name} with shape {self.df.shape}")
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def generate_visualization(self, query):
        """
        Generate a visualization based on a natural language query.
        
        Args:
            query (str): The natural language query
            
        Returns:
            dict: A dictionary containing:
                - success: Boolean indicating if generation was successful
                - html: HTML content to display
                - visualization: Base64 encoded image
                - insights: List of insights about the data
                - chart_type: The type of chart generated
                - error: Error message if generation failed
        """
        if not self.processor:
            return {
                'success': False,
                'error': "No dataset loaded. Please load a dataset first."
            }
        
        try:
            # Process the query using the advanced NLP processor
            result = self.processor.process_query(query)
            
            # Extract the chart type for metadata
            chart_type = self._extract_chart_type_from_result(result)
            
            return {
                'success': True,
                'html': result['html'],
                'visualization': result['visualization'],
                'insights': result['insights'],
                'chart_type': chart_type,
                'dataset_name': self.dataset.name
            }
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            return {
                'success': False,
                'error': f"Error generating visualization: {str(e)}"
            }
    
    def generate_report(self, query):
        """
        Generate a comprehensive report based on a natural language query.
        This includes multiple visualizations and detailed insights.
        
        Args:
            query (str): The natural language query
            
        Returns:
            dict: A dictionary containing:
                - success: Boolean indicating if generation was successful
                - html: HTML content to display
                - visualizations: List of Base64 encoded images
                - insights: List of insights about the data
                - error: Error message if generation failed
        """
        if not self.processor:
            return {
                'success': False,
                'error': "No dataset loaded. Please load a dataset first."
            }
        
        try:
            # Process the main query
            main_result = self.processor.process_query(query)
            
            # Generate additional visualizations based on the query
            additional_visualizations = self._generate_additional_visualizations(query)
            
            # Combine all insights
            all_insights = main_result['insights']
            for viz in additional_visualizations:
                if 'insights' in viz and viz['insights']:
                    # Add only unique insights
                    for insight in viz['insights']:
                        if insight not in all_insights:
                            all_insights.append(insight)
            
            # Build comprehensive HTML report
            html = "<h2>Comprehensive Data Report</h2>"
            html += f"<h3>Analysis of {self.dataset.name}</h3>"
            
            # Add main visualization
            if main_result['visualization']:
                html += "<h4>Primary Visualization</h4>"
                html += f"<div class='visualization-container mt-3 mb-3'>"
                html += f"<img src='{main_result['visualization']}' class='img-fluid' alt='Primary Visualization'>"
                html += f"</div>"
            
            # Add insights section
            html += "<h4>Key Insights</h4>"
            html += "<ul class='insights-list'>"
            for insight in all_insights[:10]:  # Limit to top 10 insights
                html += f"<li>{insight}</li>"
            html += "</ul>"
            
            # Add additional visualizations
            if additional_visualizations:
                html += "<h4>Additional Visualizations</h4>"
                html += "<div class='row'>"
                for i, viz in enumerate(additional_visualizations):
                    if viz['visualization']:
                        html += "<div class='col-md-6 mb-3'>"
                        html += f"<div class='card'>"
                        html += f"<div class='card-body'>"
                        html += f"<h5 class='card-title'>{viz.get('title', f'Visualization {i+1}')}</h5>"
                        html += f"<img src='{viz['visualization']}' class='img-fluid' alt='Additional Visualization'>"
                        html += f"</div>"
                        html += f"</div>"
                        html += "</div>"
                html += "</div>"
            
            # Add data summary
            html += "<h4>Dataset Summary</h4>"
            html += "<div class='table-responsive'>"
            html += "<table class='table table-sm table-bordered'>"
            html += "<thead><tr><th>Column</th><th>Type</th><th>Summary</th></tr></thead>"
            html += "<tbody>"
            
            for col in self.df.columns[:10]:  # Limit to first 10 columns
                col_type = str(self.df[col].dtype)
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    summary = f"Min: {self.df[col].min():.2f}, Max: {self.df[col].max():.2f}, Avg: {self.df[col].mean():.2f}"
                else:
                    unique_count = self.df[col].nunique()
                    top_val = self.df[col].value_counts().index[0] if not self.df[col].empty else "N/A"
                    summary = f"Unique values: {unique_count}, Most common: {top_val}"
                
                html += f"<tr><td>{col}</td><td>{col_type}</td><td>{summary}</td></tr>"
            
            html += "</tbody></table></div>"
            
            # Collect all visualizations
            all_visualizations = [main_result['visualization']] if main_result['visualization'] else []
            for viz in additional_visualizations:
                if viz['visualization']:
                    all_visualizations.append(viz['visualization'])
            
            return {
                'success': True,
                'html': html,
                'visualizations': all_visualizations,
                'insights': all_insights,
                'dataset_name': self.dataset.name
            }
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return {
                'success': False,
                'error': f"Error generating report: {str(e)}"
            }
    
    def _generate_additional_visualizations(self, query):
        """
        Generate additional visualizations based on the main query
        
        Args:
            query (str): The original query
            
        Returns:
            list: List of visualization results
        """
        additional_viz = []
        
        try:
            # Identify the main focus of the query
            query_lower = query.lower()
            
            # Generate a distribution visualization if appropriate
            if any(term in query_lower for term in ['distribution', 'spread', 'range']):
                # Find numeric columns
                numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                if numeric_cols:
                    # Take the first mentioned column or the first numeric column
                    for col in numeric_cols:
                        if col.lower() in query_lower:
                            target_col = col
                            break
                    else:
                        target_col = numeric_cols[0]
                    
                    # Create distribution plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(self.df[target_col].dropna(), kde=True, ax=ax)
                    ax.set_title(f'Distribution of {target_col}')
                    
                    # Convert to base64
                    img_data = self._fig_to_base64(fig)
                    plt.close(fig)
                    
                    # Add insights about the distribution
                    insights = [
                        f"The distribution of {target_col} has a mean of {self.df[target_col].mean():.2f} and median of {self.df[target_col].median():.2f}.",
                        f"The range of {target_col} is from {self.df[target_col].min():.2f} to {self.df[target_col].max():.2f}."
                    ]
                    
                    # Check for skewness
                    skewness = self.df[target_col].skew()
                    if abs(skewness) > 1:
                        skew_direction = "right" if skewness > 0 else "left"
                        insights.append(f"The distribution is skewed to the {skew_direction} with a skewness of {skewness:.2f}.")
                    
                    additional_viz.append({
                        'visualization': img_data,
                        'title': f'Distribution of {target_col}',
                        'insights': insights
                    })
            
            # Generate a correlation visualization if appropriate
            if any(term in query_lower for term in ['correlation', 'relationship', 'compare']):
                # Find numeric columns
                numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                if len(numeric_cols) >= 2:
                    # Create correlation heatmap
                    corr_matrix = self.df[numeric_cols].corr()
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                    ax.set_title('Correlation Matrix')
                    
                    # Convert to base64
                    img_data = self._fig_to_base64(fig)
                    plt.close(fig)
                    
                    # Find strongest correlations
                    corr_unstack = corr_matrix.unstack()
                    corr_unstack = corr_unstack[corr_unstack < 1.0]  # Remove self-correlations
                    strongest_corr = corr_unstack.abs().sort_values(ascending=False)
                    
                    insights = []
                    if not strongest_corr.empty:
                        top_corr_idx = strongest_corr.index[0]
                        top_corr_val = strongest_corr.iloc[0]
                        corr_type = "positive" if top_corr_val > 0 else "negative"
                        insights.append(f"The strongest {corr_type} correlation is between {top_corr_idx[0]} and {top_corr_idx[1]} at {top_corr_val:.2f}.")
                    
                    additional_viz.append({
                        'visualization': img_data,
                        'title': 'Correlation Matrix',
                        'insights': insights
                    })
            
            # Generate a time series visualization if appropriate
            datetime_cols = [col for col in self.df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if datetime_cols and any(term in query_lower for term in ['time', 'trend', 'over time', 'historical']):
                # Try to convert to datetime
                date_col = datetime_cols[0]
                try:
                    # Convert to datetime if not already
                    if not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
                        self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
                    
                    # Find a numeric column to plot
                    numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    if numeric_cols:
                        # Take the first mentioned column or the first numeric column
                        for col in numeric_cols:
                            if col.lower() in query_lower:
                                target_col = col
                                break
                        else:
                            target_col = numeric_cols[0]
                        
                        # Create time series plot
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Group by date if there are many data points
                        if len(self.df) > 100:
                            # Resample to appropriate frequency
                            df_plot = self.df.copy()
                            df_plot.set_index(date_col, inplace=True)
                            df_plot = df_plot.resample('D').mean()  # Daily average
                            df_plot[target_col].plot(ax=ax)
                        else:
                            self.df.plot(x=date_col, y=target_col, ax=ax)
                        
                        ax.set_title(f'{target_col} Over Time')
                        ax.set_xlabel('Date')
                        ax.set_ylabel(target_col)
                        
                        # Convert to base64
                        img_data = self._fig_to_base64(fig)
                        plt.close(fig)
                        
                        # Add insights about the time series
                        insights = []
                        
                        # Check for trend
                        if len(self.df) > 2:
                            first_val = self.df[target_col].iloc[0]
                            last_val = self.df[target_col].iloc[-1]
                            change = last_val - first_val
                            pct_change = (change / first_val) * 100 if first_val != 0 else 0
                            
                            trend_direction = "increased" if change > 0 else "decreased"
                            insights.append(f"{target_col} has {trend_direction} by {abs(pct_change):.2f}% over the time period.")
                        
                        additional_viz.append({
                            'visualization': img_data,
                            'title': f'{target_col} Over Time',
                            'insights': insights
                        })
                except Exception as e:
                    logger.warning(f"Could not create time series visualization: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error generating additional visualizations: {str(e)}")
        
        return additional_viz
    
    def _extract_chart_type_from_result(self, result):
        """
        Extract the chart type from the result metadata
        
        Args:
            result (dict): The result from the NLP processor
            
        Returns:
            str: The chart type or 'unknown'
        """
        if 'metadata' in result and 'viz_type' in result['metadata'] and result['metadata']['viz_type']:
            return result['metadata']['viz_type']
        return 'unknown'
    
    def _fig_to_base64(self, fig):
        """
        Convert a matplotlib figure to a base64 encoded string
        
        Args:
            fig: The matplotlib figure
            
        Returns:
            str: Base64 encoded string
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"