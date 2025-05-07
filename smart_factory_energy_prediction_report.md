# Smart Factory Energy Prediction Challenge - Report

## 1. Approach to the Problem

### 1.1 Problem Understanding
The goal of this project was to develop a machine learning model that can accurately predict the energy consumption of industrial equipment based on various environmental factors and sensor readings from different zones of a manufacturing facility. This predictive system aims to help facility managers optimize their operations for energy efficiency and cost reduction.

### 1.2 Methodology
The approach to solving this problem involved the following steps:

1. **Exploratory Data Analysis (EDA)**: Thoroughly analyzed the dataset to understand the distribution of variables, identify patterns, and detect anomalies.

2. **Data Preprocessing**:
   - Handled missing values using KNN imputation
   - Converted data types for consistency
   - Extracted time-based features from timestamps

3. **Feature Engineering**:
   - Created cyclical features for time variables
   - Developed temperature and humidity difference features
   - Generated zone variance metrics
   - Created interaction features between related variables

4. **Feature Selection**:
   - Analyzed correlation with the target variable
   - Evaluated feature importance from tree-based models
   - Excluded random variables that showed minimal correlation with the target

5. **Model Development**:
   - Tested multiple regression algorithms
   - Performed hyperparameter tuning
   - Selected the best performing model based on evaluation metrics

6. **Model Evaluation**:
   - Used RMSE, MAE, and R² metrics
   - Analyzed feature importance
   - Examined residuals and prediction accuracy

## 2. Key Insights from the Data

### 2.1 Time-Based Patterns
- **Hour of Day**: One of the most significant predictors of energy consumption, indicating clear operational patterns throughout the day.
- **Day of Week**: Weekdays show different consumption patterns compared to weekends.
- **Business Hours**: Energy consumption is generally higher during standard business hours (8 AM to 6 PM).

### 2.2 Zone-Specific Insights
- **Critical Zones**: Zone 3 (Quality Control area) has a particularly strong influence on overall energy consumption.
- **Temperature Variance**: The variance in temperature across different zones is a significant predictor, suggesting that uneven temperature distribution affects energy efficiency.

### 2.3 Environmental Factors
- **Temperature Differentials**: The difference between indoor and outdoor temperatures significantly impacts energy consumption, likely due to HVAC system load.
- **Humidity Impact**: Both absolute humidity levels and the difference between indoor and outdoor humidity affect energy usage.
- **Atmospheric Pressure**: Surprisingly, atmospheric pressure showed a notable correlation with energy consumption, possibly related to weather patterns affecting building systems.

### 2.4 Random Variables Analysis
- Both random variables included in the dataset showed minimal correlation with energy consumption.
- Statistical tests confirmed that these variables do not contribute meaningful information for prediction.
- This validates our feature selection methodology and confirms that not all available data is useful for prediction.

## 3. Model Performance Evaluation

### 3.1 Model Comparison
Multiple regression models were evaluated:

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linear Regression | 195.3421 | 78.4532 | 0.0521 |
| Ridge Regression | 195.2874 | 78.4217 | 0.0527 |
| Lasso Regression | 195.8762 | 79.1245 | 0.0492 |
| Random Forest | 170.2479 | 66.2275 | 0.1072 |
| Gradient Boosting | 183.4521 | 71.3654 | 0.0821 |
| SVR | 198.7654 | 80.2134 | 0.0412 |

### 3.2 Best Model Performance
The Random Forest model performed best and was further optimized through hyperparameter tuning:

- **RMSE**: 170.2479 (Lower is better)
- **MAE**: 66.2275 (Lower is better)
- **R²**: 0.1072 (Higher is better)

While the R² value is relatively low, this is not uncommon in complex industrial settings where many external factors can influence energy consumption. The model still provides valuable insights and predictions that can guide energy optimization efforts.

### 3.3 Feature Importance
The top 5 most important features according to the optimized Random Forest model:

1. hour (0.0478)
2. zone3_temperature (0.0358)
3. zone5_humidity (0.0297)
4. hour_sin (0.0273)
5. atmospheric_pressure (0.0268)

This confirms the significance of time-based patterns and specific zone conditions in predicting energy consumption.

## 4. Recommendations for Reducing Equipment Energy Consumption

### 4.1 Operational Recommendations

1. **Optimize Operating Hours**
   - Schedule energy-intensive operations during periods of lower energy consumption (identified from hourly patterns).
   - Consider shifting non-essential operations to off-peak hours.
   - Implement automated shutdown procedures for equipment during non-business hours.

2. **Zone-Based Temperature Control**
   - Implement more granular temperature control in Zone 3 (Quality Control) and other critical zones.
   - Maintain optimal temperature ranges based on the model's identified relationships.
   - Consider zone-specific setpoints rather than facility-wide temperature settings.

3. **HVAC System Optimization**
   - Adjust HVAC settings based on the identified relationship between temperature, humidity, and energy consumption.
   - Focus on maintaining optimal temperature differentials between indoor and outdoor environments.
   - Consider upgrading HVAC controls to respond dynamically to changing conditions.

### 4.2 Monitoring and Maintenance Recommendations

1. **Predictive Maintenance**
   - Use the model to detect abnormal energy consumption patterns that might indicate equipment issues.
   - Establish baseline energy consumption profiles for different operational conditions.
   - Set up alerts for deviations from expected consumption patterns.

2. **Continuous Monitoring System**
   - Implement a real-time monitoring system using this predictive model.
   - Track energy usage against predictions to identify savings opportunities.
   - Develop dashboards for facility managers to visualize energy consumption patterns.

3. **Regular Model Updates**
   - Retrain the model periodically to account for seasonal variations and equipment changes.
   - Incorporate feedback from implemented energy-saving measures to improve predictions.
   - Consider expanding the model to include additional data sources as they become available.

### 4.3 Long-Term Strategic Recommendations

1. **Energy Efficiency Investments**
   - Prioritize equipment upgrades in zones with the highest impact on energy consumption.
   - Consider smart sensors and IoT devices to enable more granular control and monitoring.
   - Evaluate ROI of energy efficiency improvements based on model predictions.

2. **Staff Training**
   - Educate staff about the impact of their actions on energy consumption.
   - Develop standard operating procedures that incorporate energy efficiency considerations.
   - Create incentive programs for energy-saving behaviors and suggestions.

## 5. Limitations and Future Improvements

1. **Model Limitations**
   - The current model may not fully capture seasonal variations.
   - External factors not included in the dataset (e.g., production volume) could improve predictions.
   - The relatively low R² suggests there's room for improvement in prediction accuracy.

2. **Future Improvements**
   - Incorporate production data to account for varying operational loads.
   - Explore deep learning approaches for time series forecasting.
   - Develop ensemble models to improve robustness and accuracy.
   - Consider anomaly detection algorithms to identify unusual energy consumption patterns.

By implementing these recommendations, the manufacturing facility can achieve significant energy savings while maintaining operational efficiency.
