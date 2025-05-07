import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import KNNImputer

MODEL_PATH = 'best_model_random_forest.pkl'
SCALER_PATH = 'scaler.pkl'
FEATURE_NAMES_PATH = 'feature_names.csv'

def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_names = pd.read_csv(FEATURE_NAMES_PATH, header=None).iloc[:, 0].tolist()
        return model, scaler, feature_names
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def load_sample_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    """
    Preprocess the data for prediction.
    
    This function applies the same preprocessing steps as used during training:
    - Converts timestamp to datetime
    - Extracts time-based features
    - Creates cyclical features
    - Engineers additional features
    - Handles missing values
    - Scales the features
    
    Args:
        data (pd.DataFrame): Raw data to preprocess
        
    Returns:
        pd.DataFrame: Preprocessed data ready for prediction
    """
    df = data.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df['is_business_hours'] = df['hour'].apply(lambda x: 1 if 8 <= x <= 18 else 0)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
        df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'timestamp':  
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    for i in range(1, 10):
        col_name = f'zone{i}_temp_diff'
        if f'zone{i}_temperature' in df.columns and 'outdoor_temperature' in df.columns:
            df[col_name] = df[f'zone{i}_temperature'] - df['outdoor_temperature']
    
    for i in range(1, 10):
        col_name = f'zone{i}_humidity_diff'
        if f'zone{i}_humidity' in df.columns and 'outdoor_humidity' in df.columns:
            df[col_name] = df[f'zone{i}_humidity'] - df['outdoor_humidity']

    zone_temps = [f'zone{i}_temperature' for i in range(1, 10) if f'zone{i}_temperature' in df.columns]
    if zone_temps:
        df['zone_temp_variance'] = df[zone_temps].var(axis=1)
        df['zone_temp_range'] = df[zone_temps].max(axis=1) - df[zone_temps].min(axis=1)
    
    
    zone_humidities = [f'zone{i}_humidity' for i in range(1, 10) if f'zone{i}_humidity' in df.columns]
    if zone_humidities:
        df['zone_humidity_variance'] = df[zone_humidities].var(axis=1)
        df['zone_humidity_range'] = df[zone_humidities].max(axis=1) - df[zone_humidities].min(axis=1)
    

    if 'outdoor_temperature' in df.columns and 'outdoor_humidity' in df.columns:
        df['temp_humidity_interaction'] = df['outdoor_temperature'] * df['outdoor_humidity']
    if 'wind_speed' in df.columns and 'outdoor_temperature' in df.columns:
        df['wind_temp_interaction'] = df['wind_speed'] * df['outdoor_temperature']
    
    
    if 'random_variable1' in df.columns:
        df = df.drop(columns=['random_variable1'])
    if 'random_variable2' in df.columns:
        df = df.drop(columns=['random_variable2'])
    
    
    if 'timestamp' in df.columns:
        timestamp_col = df['timestamp']
        df_for_imputation = df.drop(columns=['timestamp'])
    else:
        df_for_imputation = df.copy()
    
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df_for_imputation), 
                             columns=df_for_imputation.columns)
    

    if 'timestamp' in df.columns:
        df_imputed['timestamp'] = timestamp_col
    

    _, scaler, feature_names = load_model()
    

    for feature in feature_names:
        if feature not in df_imputed.columns:
            df_imputed[feature] = 0  
    

    X = df_imputed[feature_names]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    
    return X_scaled

def predict_energy_consumption(preprocessed_data):
    """
    Make predictions using the trained model.
    
    Args:
        preprocessed_data (pd.DataFrame): Preprocessed data
        
    Returns:
        np.array: Predicted energy consumption values
    """
    model, _, _ = load_model()
    if model is None:
        return None
    
    try:
        predictions = model.predict(preprocessed_data)
        return predictions
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None

def evaluate_predictions(actual, predicted):
    """
    Evaluate the predictions using multiple metrics.
    
    Args:
        actual (np.array): Actual energy consumption values
        predicted (np.array): Predicted energy consumption values
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    if actual is None or predicted is None:
        return None
    
    try:
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    except Exception as e:
        print(f"Error evaluating predictions: {e}")
        return None

if __name__ == "__main__":
    print("Smart Factory Energy Prediction Model")
    print("This script provides functions to load the trained model and make predictions.")
    print("Import this module in your code to use its functionality.")
