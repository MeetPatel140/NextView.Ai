import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import re
import os
import json
from datetime import datetime
from flask import current_app
from app.models.dataset import Dataset
import logging

logger = logging.getLogger(__name__)

class InsightsProcessor:
    """
    AI Insights Processor for dataset analysis and visualization generation.
    This class takes natural language queries and translates them into data analysis,
    reports, visualizations, and AI-powered insights.
    """
    
    def __init__(self, dataset):
        """
        Initialize the insights processor with a dataset.
        
        Args:
            dataset: The Dataset model object containing the data to analyze
        """
        self.dataset = dataset
        self.df = None
        self.load_dataset()
        
        # Define common patterns and terms for intent recognition
        self.viz_terms = {
            'bar': ['bar chart', 'bar graph', 'bar plot', 'barchart', 'bargraph'],
            'line': ['line chart', 'line graph', 'line plot', 'linechart', 'trend', 'over time'],
            'pie': ['pie chart', 'pie graph', 'piechart', 'percentage', 'proportion'],
            'scatter': ['scatter plot', 'scatter chart', 'scatterplot', 'correlation', 'relationship between'],
            'histogram': ['histogram', 'distribution'],
            'heatmap': ['heatmap', 'heat map', 'correlation matrix']
        }
        
        self.insight_terms = {
            'summary': ['summary', 'overview', 'describe', 'summarize', 'statistics'],
            'correlation': ['correlation', 'relationship', 'related', 'connection', 'associated'],
            'trend': ['trend', 'pattern', 'over time', 'historical', 'change'],
            'outlier': ['outlier', 'anomaly', 'unusual', 'extreme', 'different'],
            'prediction': ['predict', 'forecast', 'estimate', 'projection', 'future']
        }
        
        self.comparison_terms = ['compare', 'comparison', 'versus', 'vs', 'against', 'difference between']
        self.temporal_terms = ['time', 'year', 'month', 'day', 'week', 'quarter', 'annual', 'monthly', 'daily', 'trend']
        self.aggregation_terms = {
            'sum': ['sum', 'total', 'add'],
            'avg': ['average', 'mean', 'avg'],
            'count': ['count', 'number of', 'frequency'],
            'max': ['maximum', 'max', 'highest', 'top'],
            'min': ['minimum', 'min', 'lowest', 'bottom']
        }
        
    def load_dataset(self):
        """Load the dataset from its file path into a pandas DataFrame"""
        try:
            if not os.path.exists(self.dataset.file_path):
                raise FileNotFoundError(f"Dataset file not found at {self.dataset.file_path}")
            
            self.df = pd.read_excel(self.dataset.file_path)
            logger.info(f"Dataset loaded with shape: {self.df.shape}")
            
            # Analyze columns for later use
            self.analyze_columns()
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def analyze_columns(self):
        """Analyze the dataset columns to identify data types and potential uses"""
        self.numeric_columns = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_columns = []
        
        # Try to identify date columns among object columns
        for col in self.categorical_columns:
            # Skip columns with too many unique values (likely not dates)
            if self.df[col].nunique() > min(100, len(self.df) / 2):
                continue
                
            # Try to parse as datetime
            try:
                sample = self.df[col].dropna().iloc[0] if not self.df[col].dropna().empty else None
                if sample and pd.to_datetime(sample, errors='coerce') is not pd.NaT:
                    self.datetime_columns.append(col)
            except:
                pass
        
        # Remove identified datetime columns from categorical
        self.categorical_columns = [col for col in self.categorical_columns if col not in self.datetime_columns]
        
        logger.info(f"Column analysis - Numeric: {len(self.numeric_columns)}, "
                   f"Categorical: {len(self.categorical_columns)}, "
                   f"DateTime: {len(self.datetime_columns)}")
    
    def process_query(self, query):
        """
        Process a natural language query and return appropriate insights, visualization, or report.
        
        Args:
            query (str): The natural language query from the user
            
        Returns:
            dict: A dictionary containing the insights response with the following keys:
                - html: HTML content for displaying in the chat
                - visualization: Base64 encoded image if a visualization was generated
                - insights: List of key insights extracted from the data
                - metadata: Additional metadata about the processing
        """
        query = query.lower()
        logger.info(f"Processing insights query: {query}")
        
        try:
            # Determine the type of insight requested
            insight_type = self._determine_insight_type(query)
            
            # Generate appropriate response based on insight type
            if insight_type == 'visualization':
                return self._generate_visualization(query)
            elif insight_type == 'summary':
                return self._generate_summary(query)
            elif insight_type == 'correlation':
                return self._generate_correlation_analysis(query)
            elif insight_type == 'trend':
                return self._generate_trend_analysis(query)
            elif insight_type == 'comparison':
                return self._generate_comparison_analysis(query)
            elif insight_type == 'outlier':
                return self._generate_outlier_analysis(query)
            else:
                # General analysis / fallback
                return self._generate_general_analysis(query)
                
        except Exception as e:
            logger.error(f"Error processing insights query: {str(e)}")
            return {
                'html': f"<p>I encountered an error while processing your query: {str(e)}</p>",
                'visualization': None,
                'insights': ["Error processing query"],
                'metadata': {'error': str(e)}
            }
    
    def _determine_insight_type(self, query):
        """Determine the type of insight requested based on the query"""
        # Check for visualization terms
        for viz_type, terms in self.viz_terms.items():
            if any(term in query for term in terms):
                return 'visualization'
        
        # Check for insight terms
        for insight_type, terms in self.insight_terms.items():
            if any(term in query for term in terms):
                return insight_type
        
        # Check for comparison terms
        if any(term in query for term in self.comparison_terms):
            return 'comparison'
        
        # Default to general analysis
        return 'general'
    
    def _generate_visualization(self, query):
        """Generate a visualization based on the query"""
        # Determine visualization type
        viz_type = 'bar'  # Default
        for vtype, terms in self.viz_terms.items():
            if any(term in query for term in terms):
                viz_type = vtype
                break
        
        # Try to identify columns mentioned in the query
        columns = self._find_matching_columns(query)
        
        if not columns:
            # If no columns mentioned, use the first numeric column
            if self.numeric_columns:
                columns = [self.numeric_columns[0]]
            else:
                return {
                    'html': "<p>I couldn't identify which columns to visualize. Please specify column names in your query.</p>",
                    'visualization': None,
                    'insights': ["No columns identified for visualization"],
                    'metadata': {'viz_type': viz_type}
                }
        
        # Generate the visualization
        plt.figure(figsize=(10, 6))
        
        if viz_type == 'bar':
            if len(columns) == 1 and columns[0] in self.numeric_columns:
                # If only one numeric column, create a frequency bar chart
                self.df[columns[0]].value_counts().sort_index().plot(kind='bar')
                plt.xlabel(columns[0])
                plt.ylabel('Count')
                plt.title(f'Distribution of {columns[0]}')
            elif len(columns) == 2 and columns[0] in self.categorical_columns and columns[1] in self.numeric_columns:
                # If one categorical and one numeric, group by categorical and aggregate numeric
                self.df.groupby(columns[0])[columns[1]].mean().plot(kind='bar')
                plt.xlabel(columns[0])
                plt.ylabel(f'Average {columns[1]}')
                plt.title(f'Average {columns[1]} by {columns[0]}')
            else:
                # Default case
                self.df[columns].plot(kind='bar')
                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.title(f'Bar Chart of {", ".join(columns)}')
        
        elif viz_type == 'line':
            if len(columns) == 1 and columns[0] in self.numeric_columns:
                # Single numeric column line chart
                self.df[columns[0]].plot(kind='line')
                plt.xlabel('Index')
                plt.ylabel(columns[0])
                plt.title(f'Line Chart of {columns[0]}')
            else:
                # Multiple columns
                self.df[columns].plot(kind='line')
                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.title(f'Line Chart of {", ".join(columns)}')
        
        elif viz_type == 'pie':
            if len(columns) == 1 and columns[0] in self.categorical_columns:
                # Create pie chart of categorical column
                self.df[columns[0]].value_counts().plot(kind='pie', autopct='%1.1f%%')
                plt.ylabel('')
                plt.title(f'Distribution of {columns[0]}')
            elif len(columns) == 2 and columns[0] in self.categorical_columns and columns[1] in self.numeric_columns:
                # Pie chart of numeric values grouped by category
                self.df.groupby(columns[0])[columns[1]].sum().plot(kind='pie', autopct='%1.1f%%')
                plt.ylabel('')
                plt.title(f'Distribution of {columns[1]} by {columns[0]}')
            else:
                # Default case
                self.df[columns[0]].value_counts().head(10).plot(kind='pie', autopct='%1.1f%%')
                plt.ylabel('')
                plt.title(f'Top 10 Values of {columns[0]}')
        
        elif viz_type == 'scatter':
            if len(columns) >= 2 and columns[0] in self.numeric_columns and columns[1] in self.numeric_columns:
                # Scatter plot of two numeric columns
                self.df.plot(kind='scatter', x=columns[0], y=columns[1])
                plt.xlabel(columns[0])
                plt.ylabel(columns[1])
                plt.title(f'Scatter Plot of {columns[1]} vs {columns[0]}')
            else:
                # Default case
                return {
                    'html': "<p>Scatter plots require at least two numeric columns. Please specify two numeric columns in your query.</p>",
                    'visualization': None,
                    'insights': ["Insufficient numeric columns for scatter plot"],
                    'metadata': {'viz_type': viz_type}
                }
        
        elif viz_type == 'histogram':
            if len(columns) == 1 and columns[0] in self.numeric_columns:
                # Histogram of numeric column
                self.df[columns[0]].plot(kind='hist', bins=20)
                plt.xlabel(columns[0])
                plt.ylabel('Frequency')
                plt.title(f'Histogram of {columns[0]}')
            else:
                # Default case
                return {
                    'html': "<p>Histograms require a numeric column. Please specify a numeric column in your query.</p>",
                    'visualization': None,
                    'insights': ["No numeric column identified for histogram"],
                    'metadata': {'viz_type': viz_type}
                }
        
        elif viz_type == 'heatmap':
            if len(self.numeric_columns) >= 2:
                # Correlation heatmap of numeric columns
                sns.heatmap(self.df[self.numeric_columns].corr(), annot=True, cmap='coolwarm')
                plt.title('Correlation Heatmap')
            else:
                # Default case
                return {
                    'html': "<p>Heatmaps require at least two numeric columns. Your dataset doesn't have enough numeric columns.</p>",
                    'visualization': None,
                    'insights': ["Insufficient numeric columns for heatmap"],
                    'metadata': {'viz_type': viz_type}
                }
        
        # Save the figure to a base64 encoded string
        buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # Generate insights about the visualization
        insights = self._generate_insights_for_visualization(viz_type, columns)
        
        # Create HTML response
        html = f"<h3>Visualization: {viz_type.capitalize()} Chart</h3>"
        html += f"<p>Showing {viz_type} chart for {', '.join(columns)}</p>"
        html += f"<img src=\"data:image/png;base64,{image_base64}\" alt=\"{viz_type} chart\" class=\"img-fluid\">"
        
        # Add insights to HTML
        html += "<h4>Key Insights:</h4><ul>"
        for insight in insights:
            html += f"<li>{insight}</li>"
        html += "</ul>"
        
        return {
            'html': html,
            'visualization': image_base64,
            'insights': insights,
            'metadata': {
                'viz_type': viz_type,
                'columns': columns
            }
        }
    
    def _generate_insights_for_visualization(self, viz_type, columns):
        """Generate insights based on the visualization type and columns"""
        insights = []
        
        try:
            if viz_type == 'bar':
                if len(columns) == 1 and columns[0] in self.numeric_columns:
                    # Insights for numeric column bar chart
                    mean_val = self.df[columns[0]].mean()
                    max_val = self.df[columns[0]].max()
                    min_val = self.df[columns[0]].min()
                    insights.append(f"The average {columns[0]} is {mean_val:.2f}")
                    insights.append(f"The maximum {columns[0]} is {max_val:.2f}")
                    insights.append(f"The minimum {columns[0]} is {min_val:.2f}")
                elif len(columns) == 2 and columns[0] in self.categorical_columns and columns[1] in self.numeric_columns:
                    # Insights for grouped bar chart
                    grouped = self.df.groupby(columns[0])[columns[1]].mean()
                    max_category = grouped.idxmax()
                    min_category = grouped.idxmin()
                    insights.append(f"The highest average {columns[1]} is in the {max_category} category")
                    insights.append(f"The lowest average {columns[1]} is in the {min_category} category")
            
            elif viz_type == 'line':
                if len(columns) == 1 and columns[0] in self.numeric_columns:
                    # Insights for line chart
                    trend = "increasing" if self.df[columns[0]].iloc[-1] > self.df[columns[0]].iloc[0] else "decreasing"
                    insights.append(f"The overall trend for {columns[0]} is {trend}")
                    insights.append(f"The average {columns[0]} is {self.df[columns[0]].mean():.2f}")
            
            elif viz_type == 'pie':
                if len(columns) == 1 and columns[0] in self.categorical_columns:
                    # Insights for pie chart
                    value_counts = self.df[columns[0]].value_counts()
                    most_common = value_counts.index[0]
                    percentage = (value_counts.iloc[0] / value_counts.sum()) * 100
                    insights.append(f"The most common {columns[0]} is {most_common} ({percentage:.1f}%)")
                    insights.append(f"There are {value_counts.shape[0]} unique values for {columns[0]}")
            
            elif viz_type == 'scatter':
                if len(columns) >= 2 and columns[0] in self.numeric_columns and columns[1] in self.numeric_columns:
                    # Insights for scatter plot
                    correlation = self.df[columns].corr().iloc[0, 1]
                    relationship = "strong positive" if correlation > 0.7 else \
                                  "moderate positive" if correlation > 0.3 else \
                                  "weak positive" if correlation > 0 else \
                                  "strong negative" if correlation < -0.7 else \
                                  "moderate negative" if correlation < -0.3 else \
                                  "weak negative" if correlation < 0 else "no"
                    insights.append(f"There is a {relationship} correlation ({correlation:.2f}) between {columns[0]} and {columns[1]}")
            
            elif viz_type == 'histogram':
                if len(columns) == 1 and columns[0] in self.numeric_columns:
                    # Insights for histogram
                    skew = self.df[columns[0]].skew()
                    skew_type = "right-skewed (positive)" if skew > 0.5 else \
                               "left-skewed (negative)" if skew < -0.5 else "approximately symmetric"
                    insights.append(f"The distribution of {columns[0]} is {skew_type}")
                    insights.append(f"The average {columns[0]} is {self.df[columns[0]].mean():.2f}")
                    insights.append(f"The median {columns[0]} is {self.df[columns[0]].median():.2f}")
            
            elif viz_type == 'heatmap':
                # Insights for heatmap
                corr_matrix = self.df[self.numeric_columns].corr()
                # Find highest correlation (excluding self-correlations)
                np.fill_diagonal(corr_matrix.values, 0)
                max_corr = corr_matrix.max().max()
                max_corr_cols = np.where(corr_matrix.values == max_corr)
                if len(max_corr_cols[0]) > 0:
                    col1 = corr_matrix.index[max_corr_cols[0][0]]
                    col2 = corr_matrix.columns[max_corr_cols[1][0]]
                    insights.append(f"The strongest correlation is between {col1} and {col2} ({max_corr:.2f})")
        
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            insights.append("Could not generate detailed insights due to an error")
        
        return insights
    
    def _generate_summary(self, query):
        """Generate a summary of the dataset"""
        # Basic dataset statistics
        num_rows = len(self.df)
        num_cols = len(self.df.columns)
        num_numeric = len(self.numeric_columns)
        num_categorical = len(self.categorical_columns)
        num_datetime = len(self.datetime_columns)
        
        # Create HTML summary
        html = f"<h3>Dataset Summary: {self.dataset.name}</h3>"
        html += f"<p>This dataset contains <strong>{num_rows}</strong> rows and <strong>{num_cols}</strong> columns.</p>"
        
        # Column breakdown
        html += "<h4>Column Types:</h4>"
        html += "<ul>"
        html += f"<li><strong>{num_numeric}</strong> numeric columns</li>"
        html += f"<li><strong>{num_categorical}</strong> categorical columns</li>"
        html += f"<li><strong>{num_datetime}</strong> datetime columns</li>"
        html += "</ul>"
        
        # Basic statistics for numeric columns
        if self.numeric_columns:
            html += "<h4>Numeric Column Statistics:</h4>"
            stats_df = self.df[self.numeric_columns].describe().round(2)
            html += stats_df.to_html(classes="table table-striped table-bordered")
        
        # Top values for categorical columns (limit to first 5 categorical columns)
        if self.categorical_columns:
            html += "<h4>Top Values in Categorical Columns:</h4>"
            for col in self.categorical_columns[:5]:  # Limit to first 5 categorical columns
                value_counts = self.df[col].value_counts().head(5)
                html += f"<p><strong>{col}</strong>: "
                html += ", ".join([f"{val} ({count})" for val, count in zip(value_counts.index, value_counts.values)])
                html += "</p>"
        
        # Generate insights
        insights = [
            f"The dataset contains {num_rows} rows and {num_cols} columns",
            f"There are {num_numeric} numeric columns, {num_categorical} categorical columns, and {num_datetime} datetime columns"
        ]
        
        # Add insights about missing values
        missing_values = self.df.isnull().sum()
        cols_with_missing = missing_values[missing_values > 0]
        if not cols_with_missing.empty:
            insights.append(f"There are {len(cols_with_missing)} columns with missing values")
            most_missing = cols_with_missing.idxmax()
            insights.append(f"The column with the most missing values is {most_missing} ({missing_values[most_missing]} missing)")
        else:
            insights.append("The dataset has no missing values")
        
        return {
            'html': html,
            'visualization': None,
            'insights': insights,
            'metadata': {
                'num_rows': num_rows,
                'num_cols': num_cols,
                'column_types': {
                    'numeric': num_numeric,
                    'categorical': num_categorical,
                    'datetime': num_datetime
                }
            }
        }
    
    def _generate_correlation_analysis(self, query):
        """Generate correlation analysis between columns"""
        if len(self.numeric_columns) < 2:
            return {
                'html': "<p>Correlation analysis requires at least two numeric columns. Your dataset doesn't have enough numeric columns.</p>",
                'visualization': None,
                'insights': ["Insufficient numeric columns for correlation analysis"],
                'metadata': {}
            }
        
        # Try to identify columns mentioned in the query
        columns = self._find_matching_columns(query, self.numeric_columns)
        
        # If no specific columns mentioned, use all numeric columns
        if not columns:
            columns = self.numeric_columns
        
        # Calculate correlation matrix
        corr_matrix = self.df[columns].corr()
        
        # Generate heatmap visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        
        # Save the figure to a base64 encoded string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # Generate insights
        insights = []
        
        # Find strongest positive correlation
        pos_corr = corr_matrix.unstack()
        pos_corr = pos_corr[pos_corr < 1.0]  # Remove self-correlations
        if not pos_corr.empty:
            max_pos_corr = pos_corr.max()
            max_pos_idx = pos_corr.idxmax()
            insights.append(f"The strongest positive correlation is between {max_pos_idx[0]} and {max_pos_idx[1]} ({max_pos_corr:.2f})")
        
        # Find strongest negative correlation
        neg_corr = corr_matrix.unstack()
        neg_corr = neg_corr[neg_corr < 1.0]  # Remove self-correlations
        if not neg_corr.empty:
            min_neg_corr = neg_corr.min()
            min_neg_idx = neg_corr.idxmin()
            if min_neg_corr < 0:
                insights.append(f"The strongest negative correlation is between {min_neg_idx[0]} and {min_neg_idx[1]} ({min_neg_corr:.2f})")
        
        # Create HTML response
        html = "<h3>Correlation Analysis</h3>"
        html += f"<p>Analyzing correlations between {len(columns)} numeric columns.</p>"
        html += f"<img src=\"data:image/png;base64,{image_base64}\" alt=\"Correlation Heatmap\" class=\"img-fluid\">"
        
        # Add correlation interpretation guide
        html += "<h4>Interpretation Guide:</h4>"
        html += "<ul>"
        html += "<li><strong>1.0</strong>: Perfect positive correlation</li>"
        html += "<li><strong>0.7 to 0.9</strong>: Strong positive correlation</li>"
        html += "<li><strong>0.4 to 0.6</strong>: Moderate positive correlation</li>"
        html += "<li><strong>0.1 to 0.3</strong>: Weak positive correlation</li>"
        html += "<li><strong>0</strong>: No correlation</li>"
        html += "<li><strong>-0.1 to -0.3</strong>: Weak negative correlation</li>"
        html += "<li><strong>-0.4 to -0.6</strong>: Moderate negative correlation</li>"
        html += "<li><strong>-0.7 to -0.9</strong>: Strong negative correlation</li>"
        html += "<li><strong>-1.0</strong>: Perfect negative correlation</li>"
        html += "</ul>"
        
        # Add insights to HTML
        html += "<h4>Key Insights:</h4><ul>"
        for insight in insights:
            html += f"<li>{insight}</li>"
        html += "</ul>"
        
        return {
            'html': html,
            'visualization': image_base64,
            'insights': insights,
            'metadata': {
                'columns': columns,
                'correlation_type': 'pearson'
            }
        }
    
    def _generate_trend_analysis(self, query):
        """Generate trend analysis over time"""
        # Check if there are datetime columns
        if not self.datetime_columns:
            return {
                'html': "<p>Trend analysis requires a datetime column. Your dataset doesn't have any identified datetime columns.</p>",
                'visualization': None,
                'insights': ["No datetime columns identified for trend analysis"],
                'metadata': {}
            }
        
        # Try to identify datetime column mentioned in the query
        datetime_col = None
        for col in self.datetime_columns:
            if col.lower() in query:
                datetime_col = col
                break
        
        # If no datetime column mentioned, use the first one
        if not datetime_col:
            datetime_col = self.datetime_columns[0]
        
        # Try to identify numeric column mentioned in the query
        numeric_col = None
        for col in self.numeric_columns:
            if col.lower() in query:
                numeric_col = col
                break
        
        # If no numeric column mentioned, use the first one
        if not numeric_col and self.numeric_columns:
            numeric_col = self.numeric_columns[0]
        elif not numeric_col:
            return {
                'html': "<p>Trend analysis requires a numeric column. Your dataset doesn't have any numeric columns.</p>",
                'visualization': None,
                'insights': ["No numeric columns identified for trend analysis"],
                'metadata': {}
            }
        
        # Ensure datetime column is properly formatted
        try:
            self.df[datetime_col] = pd.to_datetime(self.df[datetime_col])
        except Exception as e:
            logger.error(f"Error converting {datetime_col} to datetime: {str(e)}")
            return {
                'html': f"<p>Error converting {datetime