"""AI Insights module for generating visualizations, reports, and insights from natural language queries.

This module enhances the chatbot with capabilities to analyze datasets and generate
visualizations, reports, and insights based on natural language queries.
"""

from flask import Blueprint

bp = Blueprint('insights', __name__)

from app.insights import routes