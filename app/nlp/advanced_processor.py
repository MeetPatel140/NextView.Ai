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
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class AdvancedNLPProcessor:
    """
    Advanced Natural Language Processing for dataset analysis and visualization generation.
    This class takes natural language queries and translates them into data analysis,
    reports, visualizations, and AI-powered insights using LlamaIndex and OpenAI.
    """
    
    def __init__(self, dataset):
        """
        Initialize the advanced NLP processor with a dataset.
        
        Args:
            dataset: The Dataset model object containing the data to analyze
        """
        self.dataset = dataset
        self.df = None
        self.load_dataset()
        
        # Initialize NLP components
        self.openai_api_key = current_app.config.get('OPENAI_API_KEY')
        self.sentence_model = None
        self.classifier = None
        self.text_generator = None
        
        # Set up NLP components
        self._setup_nlp_components()
        
        # Define common patterns and terms for intent recognition (fallback if LlamaIndex fails)
        self.viz_terms = {
            'bar': ['bar chart', 'bar graph', 'bar plot', 'barchart', 'bargraph'],
            'line': ['line chart', 'line graph', 'line plot', 'linechart', 'trend', 'over time'],
            'pie': ['pie chart', 'pie graph', 'piechart', 'percentage', 'proportion'],
            'scatter': ['scatter plot', 'scatter chart', 'scatterplot', 'correlation', 'relationship between'],
            'histogram': ['histogram', 'distribution'],
            'heatmap': ['heatmap', 'heat map', 'correlation matrix'],
            'boxplot': ['box plot', 'boxplot', 'box and whisker', 'distribution comparison'],
            'violin': ['violin plot', 'violinplot', 'distribution density'],
            'area': ['area chart', 'area plot', 'stacked area', 'cumulative'],
            'radar': ['radar chart', 'spider chart', 'web chart', 'star plot']
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
    
    def _setup_nlp_components(self):
        """Set up NLP components for text processing and analysis"""
        try:
            # Initialize the sentence transformer model for text embeddings
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize the zero-shot classification pipeline
            self.classifier = pipeline('zero-shot-classification',
                                      model='facebook/bart-large-mnli')
            
            # Initialize text generation pipeline if OpenAI key is available
            if self.openai_api_key:
                self.text_generator = pipeline('text-generation',
                                             model='gpt2')
            
            logger.info("NLP components initialized successfully")
        except Exception as e:
            logger.error(f"Error setting up NLP components: {str(e)}")
            self.classifier = None
            self.text_generator = None
    
    def process_query(self, query):
        """
        Process a natural language query and return appropriate visualization/report.
        
        Args:
            query (str): The natural language query from the user
            
        Returns:
            dict: A dictionary containing the response with the following keys:
                - html: HTML content for displaying in the chat
                - visualization: Base64 encoded image if a visualization was generated
                - insights: List of key insights extracted from the data
                - metadata: Additional metadata about the processing
        """
        query = query.lower()
        logger.info(f"Processing advanced query: {query}")
        
        try:
            # Determine query intent using zero-shot classification
            intent_labels = ['visualization', 'analysis', 'summary', 'comparison', 'prediction']
            intent_result = self.classifier(query, intent_labels)
            primary_intent = intent_result['labels'][0]
            
            # Process based on intent
            if primary_intent == 'visualization' or any(term in query for term in ['chart', 'plot', 'graph', 'visualize']):
                viz_type = self._determine_chart_type(query)
                return self._generate_visualization_response(query, viz_type)
            elif primary_intent == 'analysis':
                return self._generate_analysis_response(query)
            elif primary_intent == 'summary':
                return self._generate_summary_response()
            elif primary_intent == 'comparison':
                return self._generate_comparison_response(query)
            else:
                # Default to rule-based processing
                return self._process_with_rules(query)
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'html': f"<p>I encountered an error while processing your query: {str(e)}</p>",
                'visualization': None,
                'insights': ["Error processing query"],
                'metadata': {'error': str(e)}
            }
    
    def _process_with_llama_index(self, query):
        """Process the query using LlamaIndex agent"""
        try:
            # Determine if visualization is requested
            viz_requested = any(term in query for term in ['chart', 'plot', 'graph', 'visualize', 'visualization'])
            viz_type = self._determine_chart_type(query)
            
            # Prepare a more specific query for the agent
            if viz_requested:
                agent_query = f"Create a {viz_type} chart based on this query: {query}. First analyze the data, then provide insights, and finally describe how to visualize it."
            else:
                agent_query = f"Analyze this data based on the query: {query}. Provide detailed insights and statistics."
            
            # Get response from agent
            response = self.agent.query(agent_query)
            response_text = response.response
            
            # Extract insights from the response
            insights = self._extract_insights_from_text(response_text)
            
            # Generate visualization if requested
            visualization = None
            if viz_requested:
                visualization = self._generate_visualization_from_response(response_text, viz_type, query)
            
            # Format HTML response
            html = f"<h3>Analysis Results</h3>"
            html += f"<div class='insights-container'>{response_text}</div>"
            
            if visualization:
                html += f"<div class='visualization-container mt-3 mb-3'>"
                html += f"<img src='{visualization}' class='img-fluid' alt='Data Visualization'>"
                html += f"</div>"
            
            return {
                'html': html,
                'visualization': visualization,
                'insights': insights,
                'metadata': {'processing_method': 'llama_index', 'viz_type': viz_type if viz_requested else None}
            }
            
        except Exception as e:
            logger.error(f"Error in LlamaIndex processing: {str(e)}")
            # Fallback to rule-based processing
            return self._process_with_rules(query)
    
    def _extract_insights_from_text(self, text):
        """Extract key insights from the response text"""
        # Simple extraction by splitting on newlines and filtering
        lines = text.split('\n')
        insights = []
        
        for line in lines:
            line = line.strip()
            # Look for lines that might contain insights
            if line and len(line) > 20 and not line.startswith('#') and not line.startswith('```'):
                # Remove bullet points and numbering
                cleaned_line = re.sub(r'^[\*\-\d\.\s]+', '', line).strip()
                if cleaned_line and len(cleaned_line) > 20:
                    insights.append(cleaned_line)
        
        # Limit to top 5 insights
        return insights[:5] if insights else ["No specific insights found"]
    
    def _generate_visualization_from_response(self, response_text, viz_type, query):
        """Generate visualization based on the LlamaIndex response"""
        try:
            # Try to identify columns mentioned in the response or query
            all_text = response_text + " " + query
            columns = self._find_matching_columns(all_text)
            
            if not columns:
                # If no columns found, use the first appropriate columns based on chart type
                if viz_type in ['bar', 'pie', 'histogram'] and self.categorical_columns:
                    x_col = self.categorical_columns[0]
                    y_col = self.numeric_columns[0] if self.numeric_columns else None
                elif viz_type in ['line', 'area'] and self.datetime_columns:
                    x_col = self.datetime_columns[0]
                    y_col = self.numeric_columns[0] if self.numeric_columns else None
                elif viz_type in ['scatter', 'heatmap'] and len(self.numeric_columns) >= 2:
                    x_col = self.numeric_columns[0]
                    y_col = self.numeric_columns[1]
                else:
                    # Default fallback
                    x_col = self.df.columns[0]
                    y_col = self.df.columns[1] if len(self.df.columns) > 1 else None
            else:
                # Use identified columns
                x_col = columns[0]
                y_col = columns[1] if len(columns) > 1 else None
                if not y_col and self.numeric_columns:
                    # If only one column found and it's not numeric, find a numeric column to pair with it
                    if x_col not in self.numeric_columns:
                        y_col = self.numeric_columns[0]
                    # If the one column found is numeric, find a categorical column to pair with it
                    elif self.categorical_columns:
                        y_col, x_col = x_col, self.categorical_columns[0]
            
            # Generate the visualization
            title = f"{viz_type.capitalize()} Chart of {x_col}" + (f" vs {y_col}" if y_col else "")
            return self._create_chart_image(viz_type, self.df, x_col, y_col, title)
            
        except Exception as e:
            logger.error(f"Error generating visualization from response: {str(e)}")
            return None
    
    def _process_with_rules(self, query):
        """Process the query using rule-based approach (fallback)"""
        # Determine the type of visualization or analysis requested
        viz_requested = any(term in query for term in ['chart', 'plot', 'graph', 'visualize', 'visualization'])
        viz_type = self._determine_chart_type(query)
        
        # Find columns mentioned in the query
        columns = self._find_matching_columns(query)
        
        # If no specific columns mentioned, try to infer appropriate columns
        if not columns:
            if viz_type in ['bar', 'pie'] and self.categorical_columns:
                x_col = self.categorical_columns[0]
                y_col = self.numeric_columns[0] if self.numeric_columns else None
            elif viz_type in ['line', 'area'] and self.datetime_columns:
                x_col = self.datetime_columns[0]
                y_col = self.numeric_columns[0] if self.numeric_columns else None
            elif viz_type in ['scatter', 'heatmap'] and len(self.numeric_columns) >= 2:
                x_col = self.numeric_columns[0]
                y_col = self.numeric_columns[1]
            else:
                # Default fallback
                x_col = self.df.columns[0]
                y_col = self.df.columns[1] if len(self.df.columns) > 1 else None
        else:
            # Use identified columns
            x_col = columns[0]
            y_col = columns[1] if len(columns) > 1 else None
            if not y_col and self.numeric_columns:
                # If only one column found and it's not numeric, find a numeric column to pair with it
                if x_col not in self.numeric_columns:
                    y_col = self.numeric_columns[0]
                # If the one column found is numeric, find a categorical column to pair with it
                elif self.categorical_columns:
                    y_col, x_col = x_col, self.categorical_columns[0]
        
        # Generate basic insights
        insights = self._generate_basic_insights(x_col, y_col, viz_type)
        
        # Generate visualization if requested
        visualization = None
        if viz_requested:
            title = f"{viz_type.capitalize()} Chart of {x_col}" + (f" vs {y_col}" if y_col else "")
            visualization = self._create_chart_image(viz_type, self.df, x_col, y_col, title)
        
        # Format HTML response
        html = f"<h3>Analysis Results</h3>"
        
        if insights:
            html += "<h4>Key Insights</h4>"
            html += "<ul>"
            for insight in insights:
                html += f"<li>{insight}</li>"
            html += "</ul>"
        
        if visualization:
            html += f"<div class='visualization-container mt-3 mb-3'>"
            html += f"<img src='{visualization}' class='img-fluid' alt='Data Visualization'>"
            html += f"</div>"
        
        # Add data summary
        html += "<h4>Data Summary</h4>"
        if x_col in self.numeric_columns:
            html += f"<p><strong>{x_col}</strong>: Min: {self.df[x_col].min():.2f}, Max: {self.df[x_col].max():.2f}, Avg: {self.df[x_col].mean():.2f}</p>"
        elif x_col in self.categorical_columns:
            top_categories = self.df[x_col].value_counts().head(5)
            html += f"<p><strong>{x_col}</strong>: {len(top_categories)} unique values, most common: {', '.join(top_categories.index.astype(str))}</p>"
        
        if y_col and y_col in self.numeric_columns:
            html += f"<p><strong>{y_col}</strong>: Min: {self.df[y_col].min():.2f}, Max: {self.df[y_col].max():.2f}, Avg: {self.df[y_col].mean():.2f}</p>"
        
        return {
            'html': html,
            'visualization': visualization,
            'insights': insights,
            'metadata': {'processing_method': 'rule_based', 'viz_type': viz_type if viz_requested else None}
        }
    
    def _generate_basic_insights(self, x_col, y_col, viz_type):
        """Generate basic insights about the data"""
        insights = []
        
        try:
            # Get basic column statistics
            if x_col in self.numeric_columns:
                insights.append(f"The average {x_col} is {self.df[x_col].mean():.2f}, with values ranging from {self.df[x_col].min():.2f} to {self.df[x_col].max():.2f}.")
            elif x_col in self.categorical_columns:
                top_category = self.df[x_col].value_counts().idxmax()
                top_count = self.df[x_col].value_counts().max()
                insights.append(f"The most common {x_col} is '{top_category}' with {top_count} occurrences ({(top_count/len(self.df)*100):.1f}% of total).")
            
            if y_col and y_col in self.numeric_columns:
                insights.append(f"The average {y_col} is {self.df[y_col].mean():.2f}, with values ranging from {self.df[y_col].min():.2f} to {self.df[y_col].max():.2f}.")
            
            # Add relationship insights if both columns are specified
            if x_col and y_col:
                if x_col in self.categorical_columns and y_col in self.numeric_columns:
                    # Find category with highest average value
                    grouped = self.df.groupby(x_col)[y_col].mean().sort_values(ascending=False)
                    top_category = grouped.index[0]
                    insights.append(f"The category '{top_category}' has the highest average {y_col} at {grouped.iloc[0]:.2f}.")
                
                elif x_col in self.numeric_columns and y_col in self.numeric_columns:
                    # Calculate correlation
                    corr = self.df[[x_col, y_col]].corr().iloc[0,1]
                    if abs(corr) > 0.7:
                        relationship = "strong positive" if corr > 0 else "strong negative"
                    elif abs(corr) > 0.3:
                        relationship = "moderate positive" if corr > 0 else "moderate negative"
                    else:
                        relationship = "weak positive" if corr > 0 else "weak negative"
                    
                    insights.append(f"There is a {relationship} correlation ({corr:.2f}) between {x_col} and {y_col}.")
            
            # Add time-based insights if x_col is a datetime column
            if x_col in self.datetime_columns and y_col in self.numeric_columns:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(self.df[x_col]):
                    temp_df = self.df.copy()
                    temp_df[x_col] = pd.to_datetime(temp_df[x_col], errors='coerce')
                else:
                    temp_df = self.df
                
                # Sort by date and find trend
                temp_df = temp_df.sort_values(by=x_col)
                first_val = temp_df[y_col].iloc[0]
                last_val = temp_df[y_col].iloc[-1]
                
                if last_val > first_val * 1.1:  # 10% increase
                    insights.append(f"There is an increasing trend in {y_col} over time, with a {((last_val/first_val)-1)*100:.1f}% increase from start to end.")
                elif first_val > last_val * 1.1:  # 10% decrease
                    insights.append(f"There is a decreasing trend in {y_col} over time, with a {((first_val/last_val)-1)*100:.1f}% decrease from start to end.")
                else:
                    insights.append(f"The {y_col} values remain relatively stable over time.")
        
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            insights.append("Unable to generate detailed insights for this data.")
        
        return insights
    
    def _determine_chart_type(self, query):
        """Determine the best chart type based on the query"""
        # Check if a specific chart type is mentioned
        for chart_type, terms in self.viz_terms.items():
            if any(term in query for term in terms):
                return chart_type
        
        # Default to bar chart if no specific type mentioned
        return 'bar'
    
    def _find_matching_columns(self, query, column_list=None):
        """Find columns from the dataset that match terms in the query"""
        if column_list is None:
            column_list = self.df.columns.tolist()
            
        matches = []
        for col in column_list:
            # Convert column name to lowercase and remove underscores for matching
            col_simplified = col.lower().replace('_', ' ')
            if col_simplified in query.lower() or col.lower() in query.lower():
                matches.append(col)
                
        return matches
    
    def _create_chart_image(self, chart_type, data, x, y=None, title=None):
        """
        Create a chart image and return its base64 data URL.
        
        Args:
            chart_type: Type of chart (bar, line, pie, etc.)
            data: DataFrame with the data to plot
            x: Column name for x-axis or main dimension
            y: Column name for y-axis or values (optional for pie)
            title: Chart title
            
        Returns:
            str: Base64 data URL for the chart image
        """
        plt.figure(figsize=(10, 6))
        
        # Set seaborn style
        sns.set_style("whitegrid")
        
        # Create appropriate chart based on type
        if chart_type == 'bar':
            if y:
                ax = sns.barplot(x=x, y=y, data=data)
            else:
                # If only one column provided, create count plot
                ax = sns.countplot(x=x, data=data)
            plt.xticks(rotation=45, ha='right')
            
        elif chart_type == 'line':
            # For line charts, ensure x is sorted if it's a datetime
            if x in self.datetime_columns:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(data[x]):
                    temp_data = data.copy()
                    temp_data[x] = pd.to_datetime(temp_data[x], errors='coerce')
                    temp_data = temp_data.sort_values(by=x)
                else:
                    temp_data = data.sort_values(by=x)
                
                ax = sns.lineplot(x=x, y=y, data=temp_data, markers=True)
            else:
                ax = sns.lineplot(x=x, y=y, data=data, markers=True)
            plt.xticks(rotation=45, ha='right')
            
        elif chart_type == 'pie':
            # Prepare data for pie chart
            if y:
                pie_data = data.groupby(x)[y].sum()
            else:
                pie_data = data[x].value_counts()
            
            plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            
        elif chart_type == 'scatter':
            ax = sns.scatterplot(x=x, y=y, data=data)
            
        elif chart_type == 'histogram':
            ax = sns.histplot(data=data, x=x, kde=True)
            
        elif chart_type == 'heatmap':
            # For heatmap, we need to pivot the data if it's not already in matrix form
            if y and len(data[x].unique()) <= 20 and len(data[y].unique()) <= 20:
                # If we have a third column, use it for values, otherwise use counts
                if len(data.columns) > 2 and data.columns[2] in self.numeric_columns:
                    pivot_data = data.pivot_table(index=y, columns=x, values=data.columns[2], aggfunc='mean')
                else:
                    # Create a count-based pivot table
                    pivot_data = pd.crosstab(data[y], data[x])
                
                ax = sns.heatmap(pivot_data, annot=True, cmap="coolwarm")
            else:
                # Fallback to correlation matrix if data is not suitable for pivot
                corr_data = data.select_dtypes(include=['number']).corr()
                ax = sns.heatmap(corr_data, annot=True, cmap="coolwarm")
        
        elif chart_type == 'boxplot':
            ax = sns.boxplot(x=x, y=y, data=data)
            plt.xticks(rotation=45, ha='right')
            
        elif chart_type == 'violin':
            ax = sns.violinplot(x=x, y=y, data=data)
            plt.xticks(rotation=45, ha='right')
            
        elif chart_type == 'area':
            # For area charts, ensure x is sorted if it's a datetime
            if x in self.datetime_columns:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(data[x]):
                    temp_data = data.copy()
                    temp_data[x] = pd.to_datetime(temp_data[x], errors='coerce')
                    temp_data = temp_data.sort_values(by=x)
                else:
                    temp_data = data.sort_values(by=x)
                
                ax = sns.lineplot(x=x, y=y, data=temp_data)
                plt.fill_between(temp_data[x], temp_data[y], alpha=0.3)
            else:
                ax = sns.lineplot(x=x, y=y, data=data)
                plt.fill_between(data[x], data[y], alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            
        else:
            # Default to bar chart
            if y:
                ax = sns.barplot(x=x, y=y, data=data)
            else:
                # If only one column provided, create count plot
                ax = sns.countplot(x=x, data=data)
            plt.xticks(rotation=45, ha='right')
        
        # Add title
        if title:
            plt.title(title)
            
        plt.tight_layout()
        
        # Save to in-memory file
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        
        # Create base64 encoded data URL
        data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Close the figure to prevent memory leaks
        plt.close()
        
        return data_url