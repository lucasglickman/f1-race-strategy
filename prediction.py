import pandas as pd
import pickle

# Load the model from the pickle file
with open('strategy_model.pkl', 'rb') as file:  # make sure the path to pickle file is correct
    strategy_model = pickle.load(file)

#data cleanup function (nulls and such)
def preprocess_new_data(new_data):
    feature_columns = [
        'lap_time', 'lap_time_diff', 'tire_age_at_start', 'track_temperature',
        'air_temperature', 'wind_speed', 'humidity', 'potential_pit_stop'
    ]
    new_data['lap_time_diff'] = new_data['lap_time'].diff().fillna(0)
    new_data['potential_pit_stop'] = (new_data['lap_time_diff'] > 30).astype(int)
    X_new = new_data[feature_columns].fillna(0)
    return X_new

# compares the new practice data to the model and predicts the strategy
def predict_strategy(new_data_path):
    
    new_data = pd.read_excel('new_practice_data.xlsx')

    
    X_new = preprocess_new_data(new_data)

    
    predictions = strategy_model.predict(X_new)

    
    new_data['Predicted_Tire_Change'] = predictions
    print(new_data[['lap_number', 'Predicted_Tire_Change']].head())
    return new_data

# Example usage
predict_strategy('new_practice_data.xlsx')
result_data = predict_strategy('new_practice_data.xlsx')