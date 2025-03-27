/**
 * Chat Visualizations JavaScript
 * Handles interactive visualizations in the chat interface
 */

class ChatVisualizations {
    constructor() {
        this.chartInstances = {};
        this.initEventListeners();
    }

    /**
     * Initialize event listeners for visualization interactions
     */
    initEventListeners() {
        // Listen for visualization container clicks to toggle expanded view
        document.addEventListener('click', (event) => {
            const vizContainer = event.target.closest('.visualization-container');
            if (vizContainer && !event.target.closest('.viz-controls')) {
                this.toggleExpandedView(vizContainer);
            }
        });

        // Listen for visualization control buttons
        document.addEventListener('click', (event) => {
            // Download button
            if (event.target.closest('.viz-download-btn')) {
                const vizContainer = event.target.closest('.visualization-container');
                this.downloadVisualization(vizContainer);
            }

            // Fullscreen button
            if (event.target.closest('.viz-fullscreen-btn')) {
                const vizContainer = event.target.closest('.visualization-container');
                this.toggleFullscreen(vizContainer);
            }

            // Chart type selector
            if (event.target.closest('.viz-chart-type-selector')) {
                const selector = event.target.closest('.viz-chart-type-selector');
                const vizContainer = selector.closest('.visualization-container');
                this.changeChartType(vizContainer, selector.value);
            }
        });
    }

    /**
     * Toggle expanded view of a visualization
     * @param {HTMLElement} container - The visualization container element
     */
    toggleExpandedView(container) {
        container.classList.toggle('expanded');
        
        // If we have a Chart.js instance for this container, resize it
        const chartCanvas = container.querySelector('canvas');
        if (chartCanvas && this.chartInstances[chartCanvas.id]) {
            this.chartInstances[chartCanvas.id].resize();
        }
    }

    /**
     * Download the visualization as an image
     * @param {HTMLElement} container - The visualization container element
     */
    downloadVisualization(container) {
        const img = container.querySelector('img');
        const canvas = container.querySelector('canvas');
        
        if (img) {
            // For static images
            const link = document.createElement('a');
            link.href = img.src;
            link.download = 'visualization.png';
            link.click();
        } else if (canvas) {
            // For Chart.js visualizations
            const link = document.createElement('a');
            link.href = canvas.toDataURL('image/png');
            link.download = 'visualization.png';
            link.click();
        }
    }

    /**
     * Toggle fullscreen mode for a visualization
     * @param {HTMLElement} container - The visualization container element
     */
    toggleFullscreen(container) {
        if (!document.fullscreenElement) {
            container.requestFullscreen().catch(err => {
                console.error(`Error attempting to enable fullscreen: ${err.message}`);
            });
        } else {
            document.exitFullscreen();
        }
    }

    /**
     * Change the chart type for an interactive visualization
     * @param {HTMLElement} container - The visualization container element
     * @param {string} chartType - The new chart type
     */
    changeChartType(container, chartType) {
        const chartCanvas = container.querySelector('canvas');
        if (!chartCanvas || !this.chartInstances[chartCanvas.id]) {
            return;
        }
        
        const chart = this.chartInstances[chartCanvas.id];
        const currentData = chart.data;
        
        // Update chart type
        chart.config.type = chartType;
        chart.update();
    }

    /**
     * Initialize a Chart.js visualization
     * @param {string} canvasId - The ID of the canvas element
     * @param {Object} chartData - The data for the chart
     * @param {string} chartType - The type of chart to create
     */
    initializeChart(canvasId, chartData, chartType = 'bar') {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const chart = new Chart(ctx, {
            type: chartType,
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                    }
                }
            }
        });
        
        // Store the chart instance for later reference
        this.chartInstances[canvasId] = chart;
        return chart;
    }

    /**
     * Create interactive visualization from data
     * @param {HTMLElement} container - The container to add the visualization to
     * @param {Object} data - The data for the visualization
     */
    createInteractiveVisualization(container, data) {
        // Generate a unique ID for the canvas
        const canvasId = 'chart-' + Math.random().toString(36).substr(2, 9);
        
        // Create canvas element
        const canvas = document.createElement('canvas');
        canvas.id = canvasId;
        container.appendChild(canvas);
        
        // Initialize the chart
        this.initializeChart(canvasId, data);
        
        // Add controls
        this.addVisualizationControls(container);
    }

    /**
     * Add control buttons to a visualization container
     * @param {HTMLElement} container - The visualization container
     */
    addVisualizationControls(container) {
        const controlsDiv = document.createElement('div');
        controlsDiv.className = 'viz-controls';
        controlsDiv.innerHTML = `
            <button class="viz-download-btn" title="Download visualization">
                <i class="fas fa-download"></i>
            </button>
            <button class="viz-fullscreen-btn" title="Toggle fullscreen">
                <i class="fas fa-expand"></i>
            </button>
            <select class="viz-chart-type-selector" title="Change chart type">
                <option value="bar">Bar Chart</option>
                <option value="line">Line Chart</option>
                <option value="pie">Pie Chart</option>
                <option value="doughnut">Doughnut Chart</option>
                <option value="polarArea">Polar Area</option>
                <option value="radar">Radar Chart</option>
            </select>
        `;
        container.appendChild(controlsDiv);
    }

    /**
     * Process all visualization containers in the chat
     * Adds interactivity to static visualizations
     */
    processExistingVisualizations() {
        document.querySelectorAll('.visualization-container').forEach(container => {
            // Skip containers that already have controls
            if (container.querySelector('.viz-controls')) return;
            
            // Add controls to the container
            this.addVisualizationControls(container);
        });
    }
}

// Initialize the chat visualizations when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chatVisualizations = new ChatVisualizations();
    window.chatVisualizations.processExistingVisualizations();
});