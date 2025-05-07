# Smart Factory Energy Prediction Challenge

## Project Overview

This repository contains a solution to the Smart Factory Energy Prediction Challenge. The goal is to develop a machine learning model that can accurately predict the energy consumption of industrial equipment based on various environmental factors and sensor readings from different zones of a manufacturing facility.

## Repository Structure

```
.
├── data/                                # Contains the dataset
│   └── data.csv                         # Main dataset
├── docs/                                # Documentation files
│   └── data_description.md              # Detailed description of features
├── smart_factory_energy_prediction_complete.ipynb  # Main Jupyter notebook with full analysis
├── energy_prediction_model.py           # Python script for model deployment
├── smart_factory_energy_prediction_report.md       # Detailed report of findings and recommendations
├── best_model_random_forest.pkl         # Saved best model
├── scaler.pkl                           # Saved feature scaler
├── X_train.npy, X_test.npy              # Saved preprocessed features
├── y_train.npy, y_test.npy              # Saved target variables
├── feature_names.csv                    # List of feature names
├── feature_importance.png               # Visualization of feature importance
├── insights_and_recommendations.txt     # Summary of key insights and recommendations
└── requirements.txt                     # Required Python packages
```

## Key Components

1. **Jupyter Notebook**: A comprehensive notebook containing exploratory data analysis, data preprocessing, feature engineering, model development, evaluation, and insights.

2. **Report**: A detailed report summarizing the approach, key insights, model performance, and recommendations for reducing energy consumption.

3. **Python Script**: A script that provides functions to load the trained model and make predictions on new data.

## Key Findings

1. Time-based patterns are crucial for predicting energy consumption, with the hour of the day being one of the most important features.

2. Zone-specific factors, particularly in Zone 3 (Quality Control area), have a significant impact on energy consumption.

3. The difference between indoor and outdoor environmental conditions (temperature and humidity) strongly affects energy usage.

4. The Random Forest model performed best among the tested algorithms, suggesting that the relationship between features and energy consumption is complex and non-linear.

## Model Performance

- **RMSE**: 170.2479
- **MAE**: 66.2275
- **R²**: 0.1072

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required packages listed in requirements.txt

### Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

### Usage

To run the analysis:
1. Open the Jupyter notebook `smart_factory_energy_prediction_complete.ipynb`
2. Run all cells to reproduce the analysis

To use the trained model for predictions:
```python
import energy_prediction_model as epm

# Load sample data
data = epm.load_sample_data('path/to/data.csv')

# Preprocess the data
processed_data = epm.preprocess_data(data)

# Make predictions
predictions = epm.predict_energy_consumption(processed_data)
```

## Recommendations for Energy Reduction

1. **Optimize Operating Hours**: Schedule energy-intensive operations during periods of lower energy costs or demand.

2. **Zone-Based Temperature Control**: Implement more granular temperature control in critical zones identified by the model.

3. **Predictive Maintenance**: Use the model to detect abnormal energy consumption patterns that might indicate equipment issues.

4. **HVAC Optimization**: Adjust HVAC settings based on the identified relationship between temperature, humidity, and energy consumption.

5. **Continuous Monitoring**: Implement a real-time monitoring system using this predictive model to track energy usage and identify savings opportunities.
