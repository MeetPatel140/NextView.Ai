# NextView.AI Data Processing Module

This module provides advanced data processing capabilities for the NextView.AI platform. It handles dataset preprocessing, feature analysis, and automated insight generation to prepare data for AI analysis.

## Features

- **Automated Data Preprocessing**: Handles missing values, data type detection, and basic data cleaning
- **Feature Analysis**: Analyzes numeric, categorical, and datetime features to extract statistics and patterns
- **Correlation Analysis**: Identifies relationships between features
- **Outlier Detection**: Identifies and reports outliers in numeric features
- **Automated Insights**: Generates human-readable insights about the dataset
- **Background Processing**: Uses Celery and Redis for asynchronous processing

## API Endpoints

- `POST /data-processing/process/<dataset_id>`: Start advanced processing for a dataset
- `POST /data-processing/reprocess/<dataset_id>`: Reprocess a dataset that has already been processed
- `GET /data-processing/status/<dataset_id>`: Get the current processing status of a dataset
- `GET /data-processing/insights/<dataset_id>`: Get the insights generated during dataset processing

## Usage

```python
# Example: Start processing a dataset
response = requests.post(f"/data-processing/process/{dataset_id}")
task_id = response.json().get('task_id')

# Check processing status
status = requests.get(f"/data-processing/status/{dataset_id}").json()

# Get insights after processing is complete
insights = requests.get(f"/data-processing/insights/{dataset_id}").json()
```

## Integration with Celery

This module defines Celery tasks for background processing:

- `process_dataset_advanced`: Comprehensive dataset analysis
- `reprocess_dataset`: Re-run analysis on an existing dataset

These tasks are automatically registered with the Celery instance when the application starts.