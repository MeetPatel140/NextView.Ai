import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import re
import os
from datetime import datetime
from flask import current_app
from app.models.dataset import Dataset
import logging

logger = logging.getLogger(__name__)

class NLPProcessor:
    """
    Natural Language Processing for dataset analysis and visualization generation.
    This class takes natural language queries and translates them into data analysis,
    reports, and visualizations.
    """
    
    def __init__(self, dataset):
        """
        Initialize the NLP processor with a dataset.
        
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
        
        self.comparison_terms = ['compare', 'comparison', 'versus', 'vs', 'against', 'difference between']
        self.temporal_terms = ['time', 'year', 'month', 'day', 'week', 'quarter', 'annual', 'monthly', 'daily', 'trend']
        self.aggregation_terms = {
            'sum': ['sum', 'total', 'add'],
            'avg': ['average', 'mean', 'avg'],
            'count': ['count', 'number of', 'frequency'],
            'max': ['maximum', 'max', 'highest', 'top'],
            'min': ['minimum', 'min', 'lowest', 'bottom']
        }
        
        # Report types
        self.report_types = {
            'sales': ['sales', 'revenue', 'income', 'earnings'],
            'purchase': ['purchase', 'buy', 'acquire', 'procurement', 'expense'],
            'inventory': ['inventory', 'stock', 'storage', 'supply'],
            'customer': ['customer', 'client', 'consumer', 'buyer'],
            'financial': ['financial', 'finance', 'profit', 'loss', 'margin', 'budget']
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
        Process a natural language query and return appropriate visualization/report.
        
        Args:
            query (str): The natural language query from the user
            
        Returns:
            tuple: (html_report, base64_image)
        """
        query = query.lower()
        self.logger.info(f"Processing query: {query}")
        
        try:
            # Determine if this is a sales vs purchases comparison
            if (self._contains_any(query, ["compare", "comparison", "versus", "vs"]) and 
                self._contains_any(query, ["sales", "revenue", "income"]) and 
                self._contains_any(query, ["purchases", "costs", "expenses"])):
                self.logger.info("Generating sales vs purchases report")
                return self._generate_sales_vs_purchase_report(query)
            
            # Sales report
            elif self._contains_any(query, ["sales", "revenue", "income"]) and not self._contains_any(query, ["purchases", "costs", "expenses"]):
                self.logger.info("Generating sales report")
                return self._generate_sales_report(query)
            
            # Inventory report
            elif self._contains_any(query, ["inventory", "stock", "items"]):
                self.logger.info("Generating inventory report")
                return self._generate_inventory_report(query)
            
            # Trend analysis
            elif self._contains_any(query, ["trend", "over time", "historical", "history"]):
                self.logger.info("Generating trend analysis")
                return self._generate_trend_analysis(query)
            
            # Comparison
            elif self._contains_any(query, ["compare", "comparison", "versus", "vs"]):
                self.logger.info("Generating comparison analysis")
                return self._generate_comparison_analysis(query)
            
            # Visualization
            elif self._contains_any(query, ["chart", "plot", "graph", "show me", "visualize", "visualization"] + self.chart_types):
                self.logger.info("Generating visualization")
                return self._generate_visualization(query)
            
            # General analysis / fallback
            else:
                self.logger.info("No specific intent detected, performing general analysis")
                # Try to determine what columns are mentioned
                mentioned_columns = self._find_matching_columns(query)
                
                if mentioned_columns:
                    column = mentioned_columns[0]
                    if len(self.df[column].unique()) <= 10:  # Categorical with few unique values
                        return self._generate_visualization(f"pie chart of {column}")
                    else:
                        return self._generate_trend_analysis(f"trend of {column}")
                else:
                    # Just provide dataset overview
                    html = "<h3>Dataset Overview</h3>"
                    html += f"<p>This dataset contains {len(self.df)} rows and {len(self.df.columns)} columns.</p>"
                    
                    # Show column names and types
                    html += "<h4>Columns</h4>"
                    html += "<ul>"
                    for col in self.df.columns:
                        html += f"<li><strong>{col}</strong> ({self.df[col].dtype})</li>"
                    html += "</ul>"
                    
                    # Show sample data
                    html += "<h4>Sample Data</h4>"
                    html += self.df.head(5).to_html(classes="table table-striped table-bordered", index=False)
                    
                    return html, None
                
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return f"<p>I encountered an error while processing your query: {str(e)}</p>", None
    
    def _contains_any(self, text, terms):
        """Check if the text contains any of the given terms"""
        return any(term in text for term in terms)
    
    def _find_matching_columns(self, query, column_list=None):
        """Find columns from the dataset that match terms in the query"""
        if column_list is None:
            column_list = self.df.columns.tolist()
            
        matches = []
        for col in column_list:
            # Convert column name to lowercase and remove underscores for matching
            col_simplified = col.lower().replace('_', ' ')
            if col_simplified in query or col.lower() in query:
                matches.append(col)
                
        return matches
    
    def _determine_aggregation(self, query):
        """Determine the aggregation method from the query"""
        for agg_type, terms in self.aggregation_terms.items():
            if self._contains_any(query, terms):
                return agg_type
        # Default to sum for numeric aggregation
        return 'sum'
    
    def _determine_chart_type(self, query, x_col=None, y_col=None):
        """Determine the best chart type based on the query and data columns"""
        
        # Check if a specific chart type is mentioned
        for chart_type, terms in self.viz_terms.items():
            if self._contains_any(query, terms):
                return chart_type
        
        # If no specific chart mentioned, infer from data types
        if x_col is not None and y_col is not None:
            x_is_date = x_col in self.datetime_columns
            x_is_categorical = x_col in self.categorical_columns
            y_is_numeric = y_col in self.numeric_columns
            
            if x_is_date and y_is_numeric:
                return 'line'  # Time series
            elif x_is_categorical and y_is_numeric:
                if self.df[x_col].nunique() <= 10:
                    return 'bar'  # Few categories
                else:
                    return 'bar'  # Many categories
            elif y_is_numeric:
                if self.df[x_col].nunique() <= 5:
                    return 'pie'  # Few categories, good for pie
                else:
                    return 'bar'  # Default for categorical vs numeric
                    
        # Default to bar chart
        return 'bar'
    
    def _create_chart_image(self, chart_type, data, x, y=None, title=None, **kwargs):
        """
        Create a chart image and return its base64 data URL.
        
        Args:
            chart_type: Type of chart (bar, line, pie, etc.)
            data: DataFrame with the data to plot
            x: Column name for x-axis or main dimension
            y: Column name for y-axis or values (optional for pie)
            title: Chart title
            **kwargs: Additional parameters for the chart
            
        Returns:
            str: Base64 data URL for the chart image
        """
        plt.figure(figsize=(10, 6))
        
        # Set seaborn style
        sns.set_style("whitegrid")
        
        # Create appropriate chart based on type
        if chart_type == 'bar':
            ax = sns.barplot(x=x, y=y, data=data, **kwargs)
            plt.xticks(rotation=45, ha='right')
            
        elif chart_type == 'line':
            ax = sns.lineplot(x=x, y=y, data=data, markers=True, **kwargs)
            plt.xticks(rotation=45, ha='right')
            
        elif chart_type == 'pie':
            # Prepare data for pie chart
            pie_data = data[y].groupby(data[x]).sum()
            plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            
        elif chart_type == 'scatter':
            ax = sns.scatterplot(x=x, y=y, data=data, **kwargs)
            
        elif chart_type == 'histogram':
            ax = sns.histplot(data=data, x=x, kde=True, **kwargs)
            
        elif chart_type == 'heatmap':
            # Assuming data is a correlation matrix or similar
            ax = sns.heatmap(data, annot=True, cmap="coolwarm", **kwargs)
            
        else:
            # Default to bar chart
            ax = sns.barplot(x=x, y=y, data=data, **kwargs)
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

    def _generate_sales_vs_purchase_report(self, query):
        """Generate a sales vs. purchase comparison report"""
        logger.info("Generating sales vs purchase report")
        
        try:
            # Try to identify sales and purchase columns
            sales_columns = []
            purchase_columns = []
            
            # Look for columns with sales/revenue/income terms
            for col in self.df.columns:
                col_lower = col.lower()
                
                if any(term in col_lower for term in self.report_types['sales']):
                    sales_columns.append(col)
                    
                if any(term in col_lower for term in self.report_types['purchase']):
                    purchase_columns.append(col)
            
            # If no specific columns found, try to infer from numeric columns
            if not sales_columns and not purchase_columns and len(self.numeric_columns) >= 2:
                # Heuristic: If we have numeric columns and at least one has a positive sum
                # and at least one has a negative sum, they might be sales and purchases
                positive_sum_cols = [col for col in self.numeric_columns if self.df[col].sum() > 0]
                negative_sum_cols = [col for col in self.numeric_columns if self.df[col].sum() < 0]
                
                if positive_sum_cols:
                    sales_columns = [positive_sum_cols[0]]
                if negative_sum_cols:
                    purchase_columns = [negative_sum_cols[0]]
                    
                # If only positive values, take the first two numeric columns
                if not purchase_columns and len(positive_sum_cols) >= 2:
                    sales_columns = [positive_sum_cols[0]]
                    purchase_columns = [positive_sum_cols[1]]
            
            # If we still don't have both, use the first two numeric columns
            if (not sales_columns or not purchase_columns) and len(self.numeric_columns) >= 2:
                sales_columns = [self.numeric_columns[0]]
                purchase_columns = [self.numeric_columns[1]]
            
            if not sales_columns or not purchase_columns:
                raise ValueError("Could not identify sales and purchase columns")
            
            # Choose the first column from each category
            sales_col = sales_columns[0]
            purchase_col = purchase_columns[0]
            
            # Find a time/date column for grouping if available
            time_col = None
            if self.datetime_columns:
                time_col = self.datetime_columns[0]
            elif self.categorical_columns:
                # Look for month, year, quarter in column names
                time_related_cols = [col for col in self.categorical_columns 
                                    if any(term in col.lower() for term in 
                                         ['month', 'year', 'date', 'day', 'week', 'quarter'])]
                if time_related_cols:
                    time_col = time_related_cols[0]
                else:
                    # Use first categorical column as fallback
                    time_col = self.categorical_columns[0]
            
            # Generate analysis
            if time_col:
                # Group by time column
                grouped_data = self.df.groupby(time_col).agg({
                    sales_col: 'sum',
                    purchase_col: 'sum'
                }).reset_index()
                
                # Sort by time if possible
                try:
                    if time_col in self.datetime_columns:
                        grouped_data[time_col] = pd.to_datetime(grouped_data[time_col])
                        grouped_data = grouped_data.sort_values(time_col)
                except:
                    pass
                
                # Calculate differences and ratio
                grouped_data['Difference'] = grouped_data[sales_col] - grouped_data[purchase_col]
                grouped_data['Ratio'] = (grouped_data[sales_col] / grouped_data[purchase_col]).replace([np.inf, -np.inf], np.nan)
                
                # Generate visualization
                chart_data = {
                    'type': 'bar',
                    'config': {
                        'x_axis': time_col,
                        'y_axis': [sales_col, purchase_col],
                        'aggregation': 'sum',
                        'title': f'Sales vs. Purchases by {time_col}'
                    },
                    'image_data': self._create_comparison_chart(
                        grouped_data, time_col, sales_col, purchase_col, 
                        f'Sales vs. Purchases by {time_col}'
                    )
                }
                
                # Generate insights
                insights = []
                total_sales = grouped_data[sales_col].sum()
                total_purchases = grouped_data[purchase_col].sum()
                net_difference = total_sales - total_purchases
                
                insights.append(f"Total {sales_col}: {total_sales:,.2f}")
                insights.append(f"Total {purchase_col}: {total_purchases:,.2f}")
                insights.append(f"Net difference: {net_difference:,.2f}")
                
                if net_difference > 0:
                    insights.append(f"Sales exceed purchases by {net_difference:,.2f}")
                else:
                    insights.append(f"Purchases exceed sales by {abs(net_difference):,.2f}")
                
                # Identify top periods
                top_sales_period = grouped_data.loc[grouped_data[sales_col].idxmax()][time_col]
                top_purchase_period = grouped_data.loc[grouped_data[purchase_col].idxmax()][time_col]
                
                insights.append(f"Highest {sales_col} in {top_sales_period}")
                insights.append(f"Highest {purchase_col} in {top_purchase_period}")
                
                # Create a table for the report
                table_data = {
                    'columns': [time_col, sales_col, purchase_col, 'Difference', 'Ratio'],
                    'data': grouped_data.fillna(0).values.tolist()
                }
                
                return {
                    'explanation': f"I've analyzed the sales and purchases data grouped by {time_col}. "
                                 f"The total {sales_col} is {total_sales:,.2f} and total {purchase_col} is {total_purchases:,.2f}, "
                                 f"resulting in a net difference of {net_difference:,.2f}.",
                    'chart': chart_data,
                    'table': table_data,
                    'insights': insights,
                    'report_type': 'sales_vs_purchase'
                }
            else:
                # No time column, just aggregate totals
                total_sales = self.df[sales_col].sum()
                total_purchases = self.df[purchase_col].sum()
                difference = total_sales - total_purchases
                
                # Create simple comparison chart
                comparison_data = pd.DataFrame({
                    'Category': ['Sales', 'Purchases'],
                    'Amount': [total_sales, total_purchases]
                })
                
                chart_data = {
                    'type': 'bar',
                    'config': {
                        'x_axis': 'Category',
                        'y_axis': 'Amount',
                        'title': 'Sales vs. Purchases Comparison'
                    },
                    'image_data': self._create_chart_image(
                        'bar', comparison_data, 'Category', 'Amount', 
                        'Sales vs. Purchases Comparison'
                    )
                }
                
                insights = [
                    f"Total {sales_col}: {total_sales:,.2f}",
                    f"Total {purchase_col}: {total_purchases:,.2f}",
                    f"Net difference: {difference:,.2f}"
                ]
                
                if total_sales > total_purchases:
                    insights.append(f"Sales exceed purchases by {difference:,.2f}")
                else:
                    insights.append(f"Purchases exceed sales by {abs(difference):,.2f}")
                
                return {
                    'explanation': f"I've compared the total sales and purchases. "
                                 f"Total {sales_col} is {total_sales:,.2f} and total {purchase_col} is {total_purchases:,.2f}, "
                                 f"resulting in a net difference of {difference:,.2f}.",
                    'chart': chart_data,
                    'insights': insights,
                    'report_type': 'sales_vs_purchase'
                }
        except Exception as e:
            logger.error(f"Error generating sales vs purchase report: {str(e)}")
            return self._generate_fallback_analysis(f"I attempted to generate a sales vs. purchase report but encountered an error: {str(e)}")
    
    def _create_comparison_chart(self, data, x_col, y1_col, y2_col, title=None):
        """Create a comparison chart with two series"""
        plt.figure(figsize=(10, 6))
        
        # Create chart with two sets of bars
        x = np.arange(len(data))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, data[y1_col], width, label=y1_col)
        ax.bar(x + width/2, data[y2_col], width, label=y2_col)
        
        # Add labels and legend
        ax.set_xlabel(x_col)
        ax.set_ylabel('Amount')
        ax.set_title(title if title else f'Comparison of {y1_col} and {y2_col}')
        ax.set_xticks(x)
        ax.set_xticklabels(data[x_col], rotation=45, ha='right')
        ax.legend()
        
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
    
    def _generate_sales_report(self, query):
        """Generate a sales report"""
        logger.info("Generating sales report")
        
        try:
            # Identify sales columns
            sales_columns = []
            for col in self.df.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in self.report_types['sales']):
                    sales_columns.append(col)
            
            # If no specific sales columns found, use numeric columns as a fallback
            if not sales_columns and self.numeric_columns:
                for col in self.numeric_columns:
                    if self.df[col].sum() > 0:  # Positive sum might indicate sales/revenue
                        sales_columns.append(col)
                        break
            
            # If still no sales columns, use the first numeric column
            if not sales_columns and self.numeric_columns:
                sales_columns = [self.numeric_columns[0]]
            
            if not sales_columns:
                raise ValueError("Could not identify sales columns")
            
            # Choose the primary sales column
            sales_col = sales_columns[0]
            
            # Look for time and category dimensions
            time_col = None
            category_col = None
            
            # Try to find a time column
            if self.datetime_columns:
                time_col = self.datetime_columns[0]
            else:
                # Look for time-related terms in column names
                for col in self.categorical_columns:
                    col_lower = col.lower()
                    if any(term in col_lower for term in self.temporal_terms):
                        time_col = col
                        break
            
            # Try to find a category column (product, customer, etc.)
            if time_col:
                remaining_cat_cols = [col for col in self.categorical_columns if col != time_col]
                if remaining_cat_cols:
                    category_col = remaining_cat_cols[0]
            elif self.categorical_columns:
                category_col = self.categorical_columns[0]
                if len(self.categorical_columns) > 1:
                    time_col = self.categorical_columns[1]
            
            # Generate analysis based on available dimensions
            if time_col and category_col:
                # Time and category dimensions available
                time_series = self.df.groupby(time_col)[sales_col].sum().reset_index()
                
                # Sort by time if possible
                try:
                    time_series[time_col] = pd.to_datetime(time_series[time_col])
                    time_series = time_series.sort_values(time_col)
                except:
                    pass
                
                # Also group by category
                category_totals = self.df.groupby(category_col)[sales_col].sum().reset_index()
                category_totals = category_totals.sort_values(sales_col, ascending=False)
                
                # Top N categories
                top_n = min(5, len(category_totals))
                top_categories = category_totals.head(top_n)
                
                # Create time series chart
                time_chart = {
                    'type': 'line',
                    'config': {
                        'x_axis': time_col,
                        'y_axis': sales_col,
                        'aggregation': 'sum',
                        'title': f'{sales_col} Over Time'
                    },
                    'image_data': self._create_chart_image(
                        'line', time_series, time_col, sales_col, 
                        f'{sales_col} Over Time'
                    )
                }
                
                # Create category breakdown chart
                category_chart = {
                    'type': 'bar',
                    'config': {
                        'x_axis': category_col,
                        'y_axis': sales_col,
                        'aggregation': 'sum',
                        'title': f'Top {top_n} {category_col} by {sales_col}'
                    },
                    'image_data': self._create_chart_image(
                        'bar', top_categories, category_col, sales_col, 
                        f'Top {top_n} {category_col} by {sales_col}'
                    )
                }
                
                # Generate insights
                total_sales = self.df[sales_col].sum()
                avg_sales = self.df[sales_col].mean()
                
                top_period = time_series.loc[time_series[sales_col].idxmax()][time_col]
                bottom_period = time_series.loc[time_series[sales_col].idxmin()][time_col]
                
                top_category = category_totals.iloc[0][category_col]
                top_category_sales = category_totals.iloc[0][sales_col]
                top_category_percentage = (top_category_sales / total_sales) * 100
                
                insights = [
                    f"Total {sales_col}: {total_sales:,.2f}",
                    f"Average {sales_col}: {avg_sales:,.2f}",
                    f"Highest {sales_col} in {top_period}",
                    f"Lowest {sales_col} in {bottom_period}",
                    f"Top {category_col}: {top_category} ({top_category_percentage:.1f}% of total)"
                ]
                
                # Create a table for the report
                table_data = {
                    'columns': [time_col, sales_col],
                    'data': time_series.values.tolist()
                }
                
                # Use the time chart as the primary chart
                return {
                    'explanation': f"I've analyzed the sales data across {time_col} and {category_col}. "
                                 f"The total {sales_col} is {total_sales:,.2f} with the highest in {top_period}. "
                                 f"The top performer in {category_col} is {top_category} at {top_category_percentage:.1f}% of total sales.",
                    'chart': time_chart,
                    'table': table_data,
                    'insights': insights,
                    'report_type': 'sales'
                }
            elif time_col:
                # Only time dimension available
                time_series = self.df.groupby(time_col)[sales_col].sum().reset_index()
                
                # Sort by time if possible
                try:
                    time_series[time_col] = pd.to_datetime(time_series[time_col])
                    time_series = time_series.sort_values(time_col)
                except:
                    pass
                
                chart_data = {
                    'type': 'line',
                    'config': {
                        'x_axis': time_col,
                        'y_axis': sales_col,
                        'aggregation': 'sum',
                        'title': f'{sales_col} Over Time'
                    },
                    'image_data': self._create_chart_image(
                        'line', time_series, time_col, sales_col, 
                        f'{sales_col} Over Time'
                    )
                }
                
                # Generate insights
                total_sales = self.df[sales_col].sum()
                avg_sales = self.df[sales_col].mean()
                
                top_period = time_series.loc[time_series[sales_col].idxmax()][time_col]
                bottom_period = time_series.loc[time_series[sales_col].idxmin()][time_col]
                
                insights = [
                    f"Total {sales_col}: {total_sales:,.2f}",
                    f"Average {sales_col}: {avg_sales:,.2f}",
                    f"Highest {sales_col} in {top_period}",
                    f"Lowest {sales_col} in {bottom_period}"
                ]
                
                # Create a table for the report
                table_data = {
                    'columns': [time_col, sales_col],
                    'data': time_series.values.tolist()
                }
                
                return {
                    'explanation': f"I've analyzed the sales trends over time. "
                                 f"The total {sales_col} is {total_sales:,.2f} with an average of {avg_sales:,.2f}. "
                                 f"The highest sales were in {top_period} and the lowest in {bottom_period}.",
                    'chart': chart_data,
                    'table': table_data,
                    'insights': insights,
                    'report_type': 'sales'
                }
            elif category_col:
                # Only category dimension available
                category_totals = self.df.groupby(category_col)[sales_col].sum().reset_index()
                category_totals = category_totals.sort_values(sales_col, ascending=False)
                
                # Top N categories
                top_n = min(10, len(category_totals))
                top_categories = category_totals.head(top_n)
                
                chart_data = {
                    'type': 'bar',
                    'config': {
                        'x_axis': category_col,
                        'y_axis': sales_col,
                        'aggregation': 'sum',
                        'title': f'Top {top_n} {category_col} by {sales_col}'
                    },
                    'image_data': self._create_chart_image(
                        'bar', top_categories, category_col, sales_col, 
                        f'Top {top_n} {category_col} by {sales_col}'
                    )
                }
                
                # Generate insights
                total_sales = self.df[sales_col].sum()
                avg_sales = self.df[sales_col].mean()
                
                top_category = category_totals.iloc[0][category_col]
                top_category_sales = category_totals.iloc[0][sales_col]
                top_category_percentage = (top_category_sales / total_sales) * 100
                
                insights = [
                    f"Total {sales_col}: {total_sales:,.2f}",
                    f"Average {sales_col}: {avg_sales:,.2f}",
                    f"Top {category_col}: {top_category} ({top_category_percentage:.1f}% of total)"
                ]
                
                # Create a table for the report
                table_data = {
                    'columns': [category_col, sales_col],
                    'data': top_categories.values.tolist()
                }
                
                return {
                    'explanation': f"I've analyzed the sales breakdown by {category_col}. "
                                 f"The total {sales_col} is {total_sales:,.2f}. "
                                 f"The top performing {category_col} is {top_category} representing {top_category_percentage:.1f}% of total sales.",
                    'chart': chart_data,
                    'table': table_data,
                    'insights': insights,
                    'report_type': 'sales'
                }
            else:
                # No dimensions, just aggregates
                total_sales = self.df[sales_col].sum()
                avg_sales = self.df[sales_col].mean()
                
                # Create bar chart of overall sales
                summary_data = pd.DataFrame({'Metric': ['Total Sales'], sales_col: [total_sales]})
                
                chart_data = {
                    'type': 'bar',
                    'config': {
                        'x_axis': 'Metric',
                        'y_axis': sales_col,
                        'title': f'Total {sales_col}'
                    },
                    'image_data': self._create_chart_image(
                        'bar', summary_data, 'Metric', sales_col, 
                        f'Total {sales_col}'
                    )
                }
                
                insights = [
                    f"Total {sales_col}: {total_sales:,.2f}",
                    f"Average {sales_col}: {avg_sales:,.2f}",
                    f"Number of records: {len(self.df)}"
                ]
                
                return {
                    'explanation': f"I've analyzed the total sales. "
                                 f"The total {sales_col} is {total_sales:,.2f} with an average of {avg_sales:,.2f} "
                                 f"across {len(self.df)} records.",
                    'chart': chart_data,
                    'insights': insights,
                    'report_type': 'sales'
                }
        except Exception as e:
            logger.error(f"Error generating sales report: {str(e)}")
            return self._generate_fallback_analysis(f"I attempted to generate a sales report but encountered an error: {str(e)}")
    
    def _generate_purchase_report(self):
        """Generate a purchase report"""
        # Implementation similar to sales report but for purchase data
        # [For brevity, implementation is similar to _generate_sales_report but focused on purchase columns]
        logger.info("Generating purchase report")
        
        # Try to identify purchase columns
        purchase_columns = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in self.report_types['purchase']):
                purchase_columns.append(col)
                
        # If no purchase columns found, try numeric columns with negative values
        if not purchase_columns and self.numeric_columns:
            for col in self.numeric_columns:
                # If column has predominantly negative values, it might be purchases/expenses
                if self.df[col].min() < 0 and self.df[col].mean() < 0:
                    purchase_columns.append(col)
                    break
                    
        # If still no purchase columns, use a appropriate numeric column
        if not purchase_columns and self.numeric_columns:
            purchase_columns = [self.numeric_columns[0]]
            
        # Rest of the implementation would follow same pattern as sales report
        # For brevity, we'll delegate to the sales report with a column rename
        try:
            if not purchase_columns:
                raise ValueError("Could not identify purchase columns")
                
            # Use purchase column as the metric
            purchase_col = purchase_columns[0]
            
            # Create a copy of the DataFrame with the purchase column renamed to a generic name
            temp_df = self.df.copy()
            temp_df['Purchase_Amount'] = temp_df[purchase_col]
            
            # Temporarily swap the dataframe
            original_df = self.df
            self.df = temp_df
            
            # Generate a "sales" report but using purchase data
            result = self._generate_sales_report(query)
            
            # Restore original dataframe
            self.df = original_df
            
            # Update report type and explanations
            result['report_type'] = 'purchase'
            result['explanation'] = result['explanation'].replace('sales', 'purchase')
            
            # Update chart titles
            if 'chart' in result and 'config' in result['chart']:
                result['chart']['config']['title'] = result['chart']['config']['title'].replace('Sales', 'Purchases')
                
            return result
            
        except Exception as e:
            logger.error(f"Error generating purchase report: {str(e)}")
            return self._generate_fallback_analysis(f"I attempted to generate a purchase report but encountered an error: {str(e)}")
    
    def _generate_fallback_analysis(self, error_message=None):
        """Generate a fallback analysis when a specific report fails"""
        explanation = "I've analyzed your dataset and generated a general overview."
        if error_message:
            explanation = f"{error_message}\n\nI've created a general overview of your data instead."
            
        # Get basic statistics
        num_rows = len(self.df)
        num_cols = len(self.df.columns)
        
        # Summarize numeric columns
        numeric_summary = {}
        for col in self.numeric_columns[:5]:  # Limit to first 5 numeric columns
            numeric_summary[col] = {
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'mean': self.df[col].mean(),
                'sum': self.df[col].sum()
            }
            
        # Generate insights
        insights = [
            f"Dataset has {num_rows} rows and {num_cols} columns",
            f"There are {len(self.numeric_columns)} numeric columns and {len(self.categorical_columns)} categorical columns"
        ]
        
        for col, stats in numeric_summary.items():
            insights.append(f"{col}: Sum = {stats['sum']:,.2f}, Mean = {stats['mean']:,.2f}")
            
        # Create a simple chart of the first numeric column if available
        chart_data = None
        if self.numeric_columns and self.categorical_columns:
            try:
                x_col = self.categorical_columns[0]
                y_col = self.numeric_columns[0]
                
                # Limit categories for readability
                top_categories = self.df.groupby(x_col)[y_col].sum().nlargest(10).reset_index()
                
                chart_data = {
                    'type': 'bar',
                    'config': {
                        'x_axis': x_col,
                        'y_axis': y_col,
                        'aggregation': 'sum',
                        'title': f'{y_col} by {x_col}'
                    },
                    'image_data': self._create_chart_image(
                        'bar', top_categories, x_col, y_col, 
                        f'{y_col} by {x_col}'
                    )
                }
            except Exception as chart_err:
                logger.error(f"Error creating fallback chart: {str(chart_err)}")
                
        # Create a simple table of the first few rows
        table_data = {
            'columns': self.df.columns.tolist(),
            'data': self.df.head(5).values.tolist()
        }
        
        return {
            'explanation': explanation,
            'chart': chart_data,
            'table': table_data,
            'insights': insights,
            'report_type': 'general'
        }
    
    def _generate_general_analysis(self):
        """Generate a general data analysis"""
        return self._generate_fallback_analysis("Here's a general overview of your dataset.")
    
    def _generate_comparison_analysis(self, query):
        """Generate a comparison analysis based on the query"""
        logger.info(f"Generating comparison analysis for: {query}")
        
        try:
            # Look for specific columns mentioned in the query
            query_cols = self._find_matching_columns(query)
            
            # If no specific columns found, use the most suitable ones
            comparison_cols = []
            
            if len(query_cols) >= 2:
                comparison_cols = query_cols[:2]  # Use first two matched columns
            elif len(self.numeric_columns) >= 2:
                # Use first two numeric columns as a fallback
                comparison_cols = self.numeric_columns[:2]
            else:
                raise ValueError("Need at least two columns for comparison analysis")
                
            # Get dimensions to group by
            group_by = None
            for col in self.categorical_columns:
                if col not in comparison_cols:
                    group_by = col
                    break
            
            # Determine aggregation method
            agg_method = self._determine_aggregation(query)
            
            if group_by:
                # Group data by the selected dimension
                grouped_data = self.df.groupby(group_by)[comparison_cols].agg(agg_method).reset_index()
                
                # Sort by first comparison column for better visualization
                grouped_data = grouped_data.sort_values(comparison_cols[0], ascending=False)
                
                # Limit to top N for readability
                top_n = min(10, len(grouped_data))
                grouped_data = grouped_data.head(top_n)
                
                # Create chart
                chart_data = {
                    'type': 'bar',
                    'config': {
                        'x_axis': group_by,
                        'y_axis': comparison_cols,
                        'aggregation': agg_method,
                        'title': f'Comparison of {comparison_cols[0]} vs {comparison_cols[1]} by {group_by}'
                    },
                    'image_data': self._create_comparison_chart(
                        grouped_data, group_by, comparison_cols[0], comparison_cols[1],
                        f'Comparison of {comparison_cols[0]} vs {comparison_cols[1]} by {group_by}'
                    )
                }
                
                # Generate insights
                insights = []
                
                # Total and average for each column
                for col in comparison_cols:
                    total = self.df[col].sum()
                    avg = self.df[col].mean()
                    insights.append(f"Total {col}: {total:,.2f}")
                    insights.append(f"Average {col}: {avg:,.2f}")
                
                # Comparison ratio
                ratio = self.df[comparison_cols[0]].sum() / self.df[comparison_cols[1]].sum() if self.df[comparison_cols[1]].sum() != 0 else float('inf')
                insights.append(f"Ratio of {comparison_cols[0]} to {comparison_cols[1]}: {ratio:.2f}")
                
                # Create table for the report
                table_data = {
                    'columns': [group_by] + comparison_cols,
                    'data': grouped_data.values.tolist()
                }
                
                return {
                    'explanation': f"I've compared {comparison_cols[0]} and {comparison_cols[1]} across different {group_by} values. "
                                 f"The total for {comparison_cols[0]} is {self.df[comparison_cols[0]].sum():,.2f} and for {comparison_cols[1]} is {self.df[comparison_cols[1]].sum():,.2f}.",
                    'chart': chart_data,
                    'table': table_data,
                    'insights': insights,
                    'report_type': 'comparison'
                }
            else:
                # No grouping dimension, just compare totals
                comparison_vals = [self.df[col].sum() for col in comparison_cols]
                
                # Create simple comparison chart
                comparison_data = pd.DataFrame({
                    'Metric': comparison_cols,
                    'Value': comparison_vals
                })
                
                chart_data = {
                    'type': 'bar',
                    'config': {
                        'x_axis': 'Metric',
                        'y_axis': 'Value',
                        'title': f'Comparison of {" vs ".join(comparison_cols)}'
                    },
                    'image_data': self._create_chart_image(
                        'bar', comparison_data, 'Metric', 'Value',
                        f'Comparison of {" vs ".join(comparison_cols)}'
                    )
                }
                
                # Generate insights
                insights = []
                for col, val in zip(comparison_cols, comparison_vals):
                    insights.append(f"Total {col}: {val:,.2f}")
                
                ratio = comparison_vals[0] / comparison_vals[1] if comparison_vals[1] != 0 else float('inf')
                insights.append(f"Ratio of {comparison_cols[0]} to {comparison_cols[1]}: {ratio:.2f}")
                
                return {
                    'explanation': f"I've compared the totals for {comparison_cols[0]} and {comparison_cols[1]}. "
                                 f"The total for {comparison_cols[0]} is {comparison_vals[0]:,.2f} and for {comparison_cols[1]} is {comparison_vals[1]:,.2f}, "
                                 f"with a ratio of {ratio:.2f}.",
                    'chart': chart_data,
                    'insights': insights,
                    'report_type': 'comparison'
                }
        except Exception as e:
            logger.error(f"Error generating comparison analysis: {str(e)}")
            return self._generate_fallback_analysis(f"I attempted to generate a comparison analysis but encountered an error: {str(e)}")
    
    def _generate_visualization(self, query):
        """Generate a visualization based on the query"""
        logger.info(f"Generating visualization for: {query}")
        
        try:
            # Determine chart type from query
            chart_type = None
            for ctype, terms in self.viz_terms.items():
                if self._contains_any(query, terms):
                    chart_type = ctype
                    break
            
            # If no chart type specified, try to infer one
            if not chart_type:
                chart_type = 'bar'  # Default to bar chart
            
            # Look for columns mentioned in the query
            query_cols = self._find_matching_columns(query)
            
            # Determine x and y columns based on chart type and available columns
            x_col = None
            y_col = None
            
            if chart_type == 'pie':
                # For pie charts, need a categorical x and numeric y
                if query_cols:
                    for col in query_cols:
                        if col in self.categorical_columns and not x_col:
                            x_col = col
                        elif col in self.numeric_columns and not y_col:
                            y_col = col
                
                # Fallbacks if not enough columns specified
                if not x_col and self.categorical_columns:
                    x_col = self.categorical_columns[0]
                if not y_col and self.numeric_columns:
                    y_col = self.numeric_columns[0]
            
            elif chart_type == 'line':
                # For line charts, preferably use datetime for x-axis
                if query_cols:
                    for col in query_cols:
                        if col in self.datetime_columns and not x_col:
                            x_col = col
                        elif col in self.numeric_columns and not y_col:
                            y_col = col
                
                # Fallbacks
                if not x_col:
                    if self.datetime_columns:
                        x_col = self.datetime_columns[0]
                    elif self.categorical_columns:
                        x_col = self.categorical_columns[0]
                if not y_col and self.numeric_columns:
                    y_col = self.numeric_columns[0]
            
            else:  # Default handling for bar, scatter, etc.
                if query_cols:
                    # Try to use a categorical + numeric combination if possible
                    for col in query_cols:
                        if (col in self.categorical_columns or col in self.datetime_columns) and not x_col:
                            x_col = col
                        elif col in self.numeric_columns and not y_col:
                            y_col = col
                
                # If we didn't find appropriate columns, use fallbacks
                if not x_col:
                    if self.categorical_columns:
                        x_col = self.categorical_columns[0]
                    elif self.datetime_columns:
                        x_col = self.datetime_columns[0]
                    elif self.numeric_columns and len(self.numeric_columns) > 1:
                        x_col = self.numeric_columns[0]
                        y_col = self.numeric_columns[1]
                
                if not y_col and self.numeric_columns:
                    y_col = self.numeric_columns[0]
            
            # Ensure we have both x and y columns
            if not x_col or not y_col:
                raise ValueError("Could not determine appropriate columns for visualization")
            
            # Determine aggregation method
            agg_method = self._determine_aggregation(query)
            
            # Prepare data for the chart
            if x_col in self.categorical_columns or x_col in self.datetime_columns:
                # Group by categorical/datetime column
                chart_data = self.df.groupby(x_col)[y_col].agg(agg_method).reset_index()
                
                # For datetime columns, sort by date
                if x_col in self.datetime_columns:
                    try:
                        chart_data[x_col] = pd.to_datetime(chart_data[x_col])
                        chart_data = chart_data.sort_values(x_col)
                    except:
                        pass
                else:
                    # For categorical, sort by y value for better visualization
                    chart_data = chart_data.sort_values(y_col, ascending=False)
                
                # Limit to top N for readability
                if len(chart_data) > 15 and chart_type != 'line':
                    chart_data = chart_data.head(15)
            else:
                # Both x and y are numeric, no aggregation needed
                chart_data = self.df[[x_col, y_col]].dropna()
            
            # Create the chart
            viz_data = {
                'type': chart_type,
                'config': {
                    'x_axis': x_col,
                    'y_axis': y_col,
                    'aggregation': agg_method,
                    'title': f'{y_col} by {x_col}'
                },
                'image_data': self._create_chart_image(
                    chart_type, chart_data, x_col, y_col,
                    f'{y_col} by {x_col}'
                )
            }
            
            # Generate insights
            insights = []
            
            if chart_type == 'pie':
                # For pie charts, show distribution percentages
                total = chart_data[y_col].sum()
                top_category = chart_data.iloc[0][x_col]
                top_value = chart_data.iloc[0][y_col]
                top_percentage = (top_value / total) * 100 if total else 0
                
                insights.append(f"Total {y_col}: {total:,.2f}")
                insights.append(f"Largest segment: {top_category} ({top_percentage:.1f}%)")
                insights.append(f"Number of categories: {len(chart_data)}")
            
            elif chart_type == 'line':
                # For line charts, focus on trends
                if len(chart_data) > 1:
                    start_val = chart_data.iloc[0][y_col]
                    end_val = chart_data.iloc[-1][y_col]
                    change = end_val - start_val
                    pct_change = (change / start_val) * 100 if start_val else 0
                    
                    insights.append(f"Starting value: {start_val:,.2f}")
                    insights.append(f"Ending value: {end_val:,.2f}")
                    insights.append(f"Overall change: {change:,.2f} ({pct_change:+.1f}%)")
                    
                    # Identify peaks and troughs
                    peak = chart_data.loc[chart_data[y_col].idxmax()]
                    trough = chart_data.loc[chart_data[y_col].idxmin()]
                    
                    insights.append(f"Peak: {peak[y_col]:,.2f} at {peak[x_col]}")
                    insights.append(f"Trough: {trough[y_col]:,.2f} at {trough[x_col]}")
            
            else:  # Bar, scatter, etc.
                if x_col in self.categorical_columns:
                    # For categorical x-axis, highlight extremes
                    top_category = chart_data.iloc[0][x_col]
                    top_value = chart_data.iloc[0][y_col]
                    
                    insights.append(f"Highest {y_col}: {top_value:,.2f} for {top_category}")
                    insights.append(f"Average {y_col}: {chart_data[y_col].mean():,.2f}")
                    insights.append(f"Total {y_col}: {chart_data[y_col].sum():,.2f}")
                else:
                    # For numeric x-axis, focus on correlation or pattern
                    insights.append(f"Average {y_col}: {chart_data[y_col].mean():,.2f}")
                    insights.append(f"Range of {y_col}: {chart_data[y_col].min():,.2f} to {chart_data[y_col].max():,.2f}")
                    
                    if len(chart_data) > 5:
                        corr = chart_data[x_col].corr(chart_data[y_col])
                        insights.append(f"Correlation between {x_col} and {y_col}: {corr:.2f}")
            
            # Create a table for the report
            table_data = {
                'columns': chart_data.columns.tolist(),
                'data': chart_data.values.tolist()
            }
            
            # Create the explanation based on chart type
            if chart_type == 'pie':
                explanation = f"I've created a pie chart showing the distribution of {y_col} across different {x_col} categories. "
                explanation += f"The largest segment is {top_category} at {top_percentage:.1f}% of the total."
            elif chart_type == 'line':
                explanation = f"I've created a line chart showing {y_col} over {x_col}. "
                if len(chart_data) > 1:
                    direction = "increased" if change > 0 else "decreased"
                    explanation += f"Overall, the values have {direction} by {abs(change):,.2f} ({abs(pct_change):.1f}%)."
            elif chart_type == 'bar':
                explanation = f"I've created a bar chart showing {y_col} for each {x_col}. "
                if x_col in self.categorical_columns:
                    explanation += f"The highest value is {top_value:,.2f} for {top_category}."
            elif chart_type == 'scatter':
                explanation = f"I've created a scatter plot showing the relationship between {x_col} and {y_col}. "
                if len(chart_data) > 5:
                    corr_desc = "strong positive" if corr > 0.7 else "moderate positive" if corr > 0.3 else "weak positive" if corr > 0 else "strong negative" if corr < -0.7 else "moderate negative" if corr < -0.3 else "weak negative" if corr < 0 else "no"
                    explanation += f"There appears to be a {corr_desc} correlation between these variables."
            else:
                explanation = f"I've created a {chart_type} chart using {x_col} and {y_col}."
            
            return {
                'explanation': explanation,
                'chart': viz_data,
                'table': table_data,
                'insights': insights,
                'report_type': 'visualization'
            }
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            return self._generate_fallback_analysis(f"I attempted to generate a visualization but encountered an error: {str(e)}")
    
    def _generate_trend_analysis(self, query):
        """Generate a trend analysis over time"""
        logger.info(f"Generating trend analysis for: {query}")
        
        try:
            # Look for a time-related column
            time_col = None
            if self.datetime_columns:
                time_col = self.datetime_columns[0]
            else:
                # Look for columns with time-related names
                for col in self.categorical_columns:
                    col_lower = col.lower()
                    if any(term in col_lower for term in self.temporal_terms):
                        time_col = col
                        break
            
            if not time_col:
                raise ValueError("Could not find a suitable time column for trend analysis")
            
            # Look for metric columns mentioned in the query
            metric_cols = self._find_matching_columns(query, self.numeric_columns)
            
            # If no specific metric columns found, use the first numeric column
            if not metric_cols and self.numeric_columns:
                metric_cols = [self.numeric_columns[0]]
            
            if not metric_cols:
                raise ValueError("Could not find a suitable metric column for trend analysis")
            
            # Use the first identified metric column
            metric_col = metric_cols[0]
            
            # Determine aggregation method
            agg_method = self._determine_aggregation(query)
            
            # Group by time column
            trend_data = self.df.groupby(time_col)[metric_col].agg(agg_method).reset_index()
            
            # Try to sort by time if possible
            try:
                if time_col in self.datetime_columns:
                    trend_data[time_col] = pd.to_datetime(trend_data[time_col])
                trend_data = trend_data.sort_values(time_col)
            except:
                # If conversion fails, try to sort intelligently
                try:
                    if time_col.lower() in ['year', 'month', 'quarter', 'week', 'day']:
                        trend_data = trend_data.sort_values(time_col)
                except:
                    pass
            
            # Create the chart
            chart_data = {
                'type': 'line',
                'config': {
                    'x_axis': time_col,
                    'y_axis': metric_col,
                    'aggregation': agg_method,
                    'title': f'{metric_col} Trend Over {time_col}'
                },
                'image_data': self._create_chart_image(
                    'line', trend_data, time_col, metric_col,
                    f'{metric_col} Trend Over {time_col}'
                )
            }
            
            # Generate insights
            insights = []
            
            if len(trend_data) > 1:
                # Calculate overall change
                start_val = trend_data.iloc[0][metric_col]
                end_val = trend_data.iloc[-1][metric_col]
                change = end_val - start_val
                pct_change = (change / start_val) * 100 if start_val else 0
                
                # Find peak and trough
                peak = trend_data.loc[trend_data[metric_col].idxmax()]
                trough = trend_data.loc[trend_data[metric_col].idxmin()]
                
                # Calculate moving averages if enough data points
                if len(trend_data) >= 3:
                    trend_data['MA3'] = trend_data[metric_col].rolling(window=3, min_periods=1).mean()
                    recent_trend = "upward" if trend_data['MA3'].iloc[-1] > trend_data['MA3'].iloc[-2] else "downward"
                
                # Add insights
                insights.append(f"Starting {metric_col}: {start_val:,.2f}")
                insights.append(f"Ending {metric_col}: {end_val:,.2f}")
                insights.append(f"Overall change: {change:,.2f} ({pct_change:+.1f}%)")
                insights.append(f"Peak: {peak[metric_col]:,.2f} at {peak[time_col]}")
                insights.append(f"Trough: {trough[metric_col]:,.2f} at {trough[time_col]}")
                
                if 'MA3' in trend_data:
                    insights.append(f"Recent trend: {recent_trend}")
            else:
                insights.append(f"Only one data point available: {trend_data.iloc[0][metric_col]:,.2f}")
            
            # Create a table for the report
            table_data = {
                'columns': [time_col, metric_col],
                'data': trend_data[[time_col, metric_col]].values.tolist()
            }
            
            # Create explanation
            explanation = f"I've analyzed the trend of {metric_col} over {time_col}. "
            
            if len(trend_data) > 1:
                direction = "increased" if change > 0 else "decreased"
                explanation += f"Overall, {metric_col} has {direction} by {abs(change):,.2f} ({abs(pct_change):.1f}%) "
                explanation += f"from {start_val:,.2f} to {end_val:,.2f}. "
                explanation += f"The highest value was {peak[metric_col]:,.2f} at {peak[time_col]} and "
                explanation += f"the lowest was {trough[metric_col]:,.2f} at {trough[time_col]}."
            else:
                explanation += f"Only one data point is available with value {trend_data.iloc[0][metric_col]:,.2f}."
            
            return {
                'explanation': explanation,
                'chart': chart_data,
                'table': table_data,
                'insights': insights,
                'report_type': 'trend'
            }
        except Exception as e:
            logger.error(f"Error generating trend analysis: {str(e)}")
            return self._generate_fallback_analysis(f"I attempted to generate a trend analysis but encountered an error: {str(e)}")
    
    # Helper methods for other reports
    def _generate_inventory_report(self, query):
        """Generate an inventory report"""
        # Simplified implementation - would typically look for stock/quantity columns
        logger.info("Generating inventory report")
        
        # Try to identify stock/inventory columns
        inventory_columns = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in self.report_types['inventory']):
                inventory_columns.append(col)
                
        # Delegate to general analysis if specific implementation not available
        if not inventory_columns:
            return self._generate_fallback_analysis("I couldn't identify specific inventory columns in this dataset. Here's a general analysis instead.")
        
        # For brevity, we'll use a simplified approach similar to sales report
        try:
            inventory_col = inventory_columns[0]
            
            # Try to find product/item column
            product_col = None
            for col in self.categorical_columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['product', 'item', 'sku', 'description']):
                    product_col = col
                    break
                    
            if not product_col and self.categorical_columns:
                product_col = self.categorical_columns[0]
                
            if product_col:
                # Group by product
                inventory_by_product = self.df.groupby(product_col)[inventory_col].sum().reset_index()
                inventory_by_product = inventory_by_product.sort_values(inventory_col, ascending=False)
                
                # Get top and bottom products
                top_n = min(10, len(inventory_by_product))
                top_products = inventory_by_product.head(top_n)
                
                # Create chart
                chart_data = {
                    'type': 'bar',
                    'config': {
                        'x_axis': product_col,
                        'y_axis': inventory_col,
                        'aggregation': 'sum',
                        'title': f'Top {top_n} Products by {inventory_col}'
                    },
                    'image_data': self._create_chart_image(
                        'bar', top_products, product_col, inventory_col,
                        f'Top {top_n} Products by {inventory_col}'
                    )
                }
                
                # Generate insights
                total_inventory = self.df[inventory_col].sum()
                avg_inventory = self.df[inventory_col].mean()
                
                top_product = inventory_by_product.iloc[0][product_col]
                top_inventory = inventory_by_product.iloc[0][inventory_col]
                
                bottom_n = min(10, len(inventory_by_product))
                bottom_products = inventory_by_product.tail(bottom_n)
                bottom_product = inventory_by_product.iloc[-1][product_col]
                bottom_inventory = inventory_by_product.iloc[-1][inventory_col]
                
                insights = [
                    f"Total {inventory_col}: {total_inventory:,.2f}",
                    f"Average {inventory_col} per product: {avg_inventory:,.2f}",
                    f"Highest inventory: {top_product} ({top_inventory:,.2f})",
                    f"Lowest inventory: {bottom_product} ({bottom_inventory:,.2f})",
                    f"Number of products: {len(inventory_by_product)}"
                ]
                
                # Create a table for the report
                table_data = {
                    'columns': [product_col, inventory_col],
                    'data': top_products.values.tolist()
                }
                
                return {
                    'explanation': f"I've analyzed the inventory levels by product. "
                                 f"The total {inventory_col} is {total_inventory:,.2f} across {len(inventory_by_product)} products. "
                                 f"The product with the highest inventory is {top_product} with {top_inventory:,.2f} units.",
                    'chart': chart_data,
                    'table': table_data,
                    'insights': insights,
                    'report_type': 'inventory'
                }
            else:
                # No product dimension, just aggregates
                total_inventory = self.df[inventory_col].sum()
                avg_inventory = self.df[inventory_col].mean()
                
                # Create summary chart
                summary_data = pd.DataFrame({'Metric': ['Total Inventory'], inventory_col: [total_inventory]})
                
                chart_data = {
                    'type': 'bar',
                    'config': {
                        'x_axis': 'Metric',
                        'y_axis': inventory_col,
                        'title': f'Total {inventory_col}'
                    },
                    'image_data': self._create_chart_image(
                        'bar', summary_data, 'Metric', inventory_col,
                        f'Total {inventory_col}'
                    )
                }
                
                insights = [
                    f"Total {inventory_col}: {total_inventory:,.2f}",
                    f"Average {inventory_col}: {avg_inventory:,.2f}",
                    f"Number of records: {len(self.df)}"
                ]
                
                return {
                    'explanation': f"I've analyzed the total inventory. "
                                 f"The total {inventory_col} is {total_inventory:,.2f} with an average of {avg_inventory:,.2f} "
                                 f"across {len(self.df)} records.",
                    'chart': chart_data,
                    'insights': insights,
                    'report_type': 'inventory'
                }
        except Exception as e:
            logger.error(f"Error generating inventory report: {str(e)}")
            return self._generate_fallback_analysis(f"I attempted to generate an inventory report but encountered an error: {str(e)}")
    
    def _generate_customer_report(self):
        """Generate a customer analysis report"""
        # Simplified implementation - similar pattern to other report types
        logger.info("Generating customer report")
        return self._generate_fallback_analysis("Customer analysis functionality is not fully implemented yet. Here's a general overview instead.") 