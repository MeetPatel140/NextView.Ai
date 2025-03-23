# NextView.AI

An AI-powered data visualization and chatbot platform for intelligent data analysis and reporting.

## Project Overview

NextView.AI is a comprehensive platform that allows users to:

- Upload Excel (.xlsx) files for data processing
- Generate smart reports with interactive tables and charts
- Apply real-time filters to visualize data
- Download customizable PDF reports
- Interact with an AI-powered chatbot for data insights and queries

## Features

### File Upload & Data Processing
- Upload Excel (.xlsx) files
- Automatic data extraction and cleaning using Pandas
- AI-powered pattern detection and data correction
- Secure data storage in MySQL database

### Smart Reports & Data Visualization
- Interactive tables and charts
- Real-time filtering, sorting, and searching
- Multiple visualization types (bar, pie, line, heatmaps, etc.)
- AI-driven data aggregation and trend analysis

### AI-Powered Chatbot
- Natural language queries about uploaded data
- AI-generated reports based on user questions
- Intelligent search capabilities
- Automated insights, correlations, and predictions

### PDF Report Generation
- Customizable PDF exports
- Combined tables and charts
- Layout customization options

### User Authentication
- Secure login and registration
- Role-based access control
- Session management

### Background Processing
- Asynchronous handling of large data files
- Background execution for AI tasks
- Real-time notifications

### API for Integrations
- RESTful API endpoints
- Webhook support
- External tool integration

## Tech Stack

### Backend
- Python (Core logic and AI)
- Flask (Web framework)
- MySQL (Database)

### Frontend
- HTML, CSS, JavaScript
- Bootstrap (UI framework)
- Chart.js (Data visualization)
- jQuery & AJAX

### AI & Data Processing
- Pandas & NumPy
- OpenAI API / LlamaIndex
- Matplotlib & Seaborn

### Other Dependencies
- Flask-SQLAlchemy
- Flask-Login
- ReportLab / WeasyPrint
- Celery & Redis

## Setup Instructions

### Prerequisites
- Python 3.8+
- MySQL
- Redis (for Celery)

### Installation

1. Clone the repository
```
git clone https://github.com/yourusername/nextview-ai.git
cd nextview-ai
```

2. Create and activate a virtual environment
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```
pip install -r requirements.txt
```

4. Configure environment variables
```
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize the database
```
flask db init
flask db migrate
flask db upgrade
```

6. Run the development server
```
flask run
```

7. In a separate terminal, start Celery worker
```
celery -A app.celery worker --loglevel=info
```

## Project Structure

```
nextview-ai/
├── app/
│   ├── __init__.py
│   ├── models/
│   ├── routes/
│   ├── services/
│   ├── static/
│   └── templates/
├── migrations/
├── tests/
├── .env
├── .gitignore
├── config.py
├── requirements.txt
└── run.py
```

## License

MIT