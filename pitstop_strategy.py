# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import random

# **Step 1: Load and preprocess practice data from multiple venues**

# Load the Excel file to get the list of venues from the sheet names
excel_file = pd.ExcelFile('mclaren_practice_2024_data.xlsx')
venues = excel_file.sheet_names

# Initialize an empty list to hold practice DataFrames
practice_dfs = []

# Loop through each venue and load the practice data
for venue in venues:
    df = pd.read_excel('mclaren_practice_2024_data.xlsx', sheet_name=venue)
    df['venue'] = venue  # Add a column to identify the venue
    practice_dfs.append(df)

# Combine all practice DataFrames into one
practice_df_combined = pd.concat(practice_dfs, ignore_index=True)

# **Load Venue Characteristics**

# Create a DataFrame containing characteristics of each venue
# Replace this with your actual venue characteristics data
venue_characteristics = pd.DataFrame({
    'venue': ['Singapore', 'Azerbaijan', 'Italy', 'Netherlands', 'Belgium', 'Hungary', 'Great Britain', 
              'Austria', 'Spain', 'Canada', 'Monaco', 'China', 'Japan', 'Australia', 'Saudi Arabia', 
              'Bahrain', 'United States'],
    'track_length_km': [5.063, 6.003, 5.793, 4.259, 7.004, 4.381, 5.891, 4.326, 4.655, 4.421, 3.337, 5.451, 5.807, 5.303, 6.174, 5.412, 5.513],
    'number_of_turns': [23, 20, 11, 14, 19, 14, 18, 10, 16, 14, 19, 16, 18, 14, 27, 15, 20],
    'average_speed_kmh': [170, 210, 250, 215, 235, 200, 240, 235, 210, 230, 165, 210, 230, 220, 250, 210, 215],
    'elevation_change_m': [9, 11, 12, 8, 100, 34, 12, 63, 26, 6, 42, 7, 40, 32, 12, 16, 30]
})

# Filter venue characteristics by the venues we have in practice data
venue_characteristics_filtered = venue_characteristics[venue_characteristics['venue'].isin(practice_df_combined['venue'].unique())]

# Merge venue characteristics with practice data
practice_df_combined = practice_df_combined.merge(venue_characteristics_filtered, on='venue', how='left')

# **Data Preprocessing for Practice Data**

# Drop rows with missing lap times
practice_df_combined.dropna(subset=['lap_time'], inplace=True)

# Forward-fill missing values
practice_df_combined.fillna(method='ffill', inplace=True)

# Feature Engineering
practice_df_combined['cumulative_lap'] = practice_df_combined.groupby(['driver_number', 'venue']).cumcount() + 1

# Define the total fuel capacity and fuel consumption per lap
TOTAL_FUEL_CAPACITY = 110  # in kg
FUEL_CONSUMPTION_PER_LAP = 2  # in kg (adjust as needed)

# Function to calculate fuel load for each lap
def calculate_fuel_load(df):
    df = df.sort_values(by=['driver_number', 'cumulative_lap'])
    df['fuel_load'] = TOTAL_FUEL_CAPACITY - (df['cumulative_lap'] - 1) * FUEL_CONSUMPTION_PER_LAP
    df['fuel_load'] = df['fuel_load'].clip(lower=0)  # Ensure fuel load doesn't go negative
    return df

# Apply the function to practice data
practice_df_combined = practice_df_combined.groupby(['driver_number', 'venue']).apply(calculate_fuel_load)

# One-Hot Encode 'tire_compound'
practice_df_combined = pd.get_dummies(practice_df_combined, columns=['tire_compound'])

# Ensure all tire compound columns are present
tire_compounds = ['tire_compound_Hard', 'tire_compound_Medium', 'tire_compound_Soft']
for comp in tire_compounds:
    if comp not in practice_df_combined.columns:
        practice_df_combined[comp] = 0

# **Define feature columns**

# Define all feature columns (excluding 'venue' one-hot encoding)
features = [
    'tire_age_at_start',
    'track_temperature',
    'air_temperature',
    'humidity',
    'cumulative_lap',
    'fuel_load',  # Add fuel_load here
    'track_length_km',
    'number_of_turns',
    'average_speed_kmh',
    'elevation_change_m',
    'tire_compound_Hard',
    'tire_compound_Medium',
    'tire_compound_Soft'
]

# **Prepare training data**

X_train = practice_df_combined[features]
y_train = practice_df_combined['lap_time']

# **Step 2: Train the predictive model on practice data**

# Initialize the model with default hyperparameters
model = RandomForestRegressor(random_state=42)

# **Step 3: Load and preprocess race data to evaluate the model's performance**

# Initialize an empty list to hold race DataFrames
race_dfs = []

# Loop through each venue and load the race data
for venue in venues:
    df = pd.read_excel('mclaren_races_2024_data.xlsx', sheet_name=venue)
    df['venue'] = venue  # Add a column to identify the venue
    race_dfs.append(df)

# Combine all race DataFrames into one
race_df_combined = pd.concat(race_dfs, ignore_index=True)

# Filter venue characteristics by the venues we have in race data
venue_characteristics_filtered = venue_characteristics[venue_characteristics['venue'].isin(race_df_combined['venue'].unique())]

# Merge venue characteristics with race data
race_df_combined = race_df_combined.merge(venue_characteristics_filtered, on='venue', how='left')

# Data Preprocessing for Race Data
race_df_combined.dropna(subset=['lap_time'], inplace=True)
race_df_combined.fillna(method='ffill', inplace=True)

# Feature Engineering
race_df_combined['cumulative_lap'] = race_df_combined.groupby(['driver_number', 'venue']).cumcount() + 1

# Apply the fuel load calculation to race data
race_df_combined = race_df_combined.groupby(['driver_number', 'venue']).apply(calculate_fuel_load)

# One-Hot Encode 'tire_compound'
race_df_combined = pd.get_dummies(race_df_combined, columns=['tire_compound'])
for comp in tire_compounds:
    if comp not in race_df_combined.columns:
        race_df_combined[comp] = 0

# **Prepare evaluation data**

X_test = race_df_combined[features]
y_test = race_df_combined['lap_time']

# **Step 4: Evaluate the model's performance and adjust hyperparameters**

# Define the hyperparameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10, 15]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Retrieve the best model
best_model = grid_search.best_estimator_

# Evaluate the best model on the race data
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Best Model Parameters: {grid_search.best_params_}")
print(f"Mean Squared Error on Race Data: {mse:.2f}")

# **Step 5: Use the optimized model to predict race outcomes based on new practice data**

# Load new practice data for the upcoming race
new_venue = 'United States'  # Replace with the new venue
practice_df_new = pd.read_excel('new_practice_data.xlsx', sheet_name=new_venue)
practice_df_new['venue'] = new_venue

# Merge venue characteristics with new practice data
practice_df_new = practice_df_new.merge(venue_characteristics, on='venue', how='left')

# Data Preprocessing
practice_df_new.dropna(subset=['lap_time'], inplace=True)
practice_df_new.fillna(method='ffill', inplace=True)

# Feature Engineering
practice_df_new['cumulative_lap'] = practice_df_new.groupby(['driver_number']).cumcount() + 1

# Apply the fuel load calculation to new practice data
practice_df_new = practice_df_new.groupby(['driver_number']).apply(calculate_fuel_load)

# One-Hot Encode 'tire_compound'
practice_df_new = pd.get_dummies(practice_df_new, columns=['tire_compound'])
for comp in tire_compounds:
    if comp not in practice_df_new.columns:
        practice_df_new[comp] = 0

# **Extract environmental conditions from new practice data**
env_conditions_new = {
    'track_temperature': practice_df_new['track_temperature'].mean(),
    'air_temperature': practice_df_new['air_temperature'].mean(),
    'humidity': practice_df_new['humidity'].mean(),
    # Include venue characteristics
    'track_length_km': practice_df_new['track_length_km'].mean(),
    'number_of_turns': practice_df_new['number_of_turns'].mean(),
    'average_speed_kmh': practice_df_new['average_speed_kmh'].mean(),
    'elevation_change_m': practice_df_new['elevation_change_m'].mean()
}

# **Define the race simulation function**

def simulate_race(model, pit_laps, tire_compounds, env_conditions, total_laps=58):
    pit_duration = 25  # seconds
    total_time = 0
    pit_stops = sorted([int(round(p)) for p in pit_laps if 1 < p < total_laps])
    
    # Define stints
    stints = []
    previous_pit = 1
    for pit in pit_stops:
        end_lap = pit - 1
        if end_lap >= previous_pit:
            stints.append((previous_pit, end_lap))
            previous_pit = pit
    if previous_pit <= total_laps:
        stints.append((previous_pit, total_laps))
    
    # Simulate each stint
    cumulative_lap = 1
    for idx, stint in enumerate(stints):
        start_lap, end_lap = stint
        laps = end_lap - start_lap + 1
        if laps <= 0:
            continue  # Skip invalid stints
        compound = tire_compounds[idx % len(tire_compounds)]
        tire_age = np.arange(1, laps + 1)
        
        # Calculate fuel load for each lap
        fuel_load = TOTAL_FUEL_CAPACITY - (np.arange(cumulative_lap, cumulative_lap + laps) - 1) * FUEL_CONSUMPTION_PER_LAP
        fuel_load = np.clip(fuel_load, a_min=0, a_max=None)
        
        # Prepare data for prediction
        stint_data = pd.DataFrame({
            'tire_age_at_start': tire_age,
            'track_temperature': env_conditions['track_temperature'],
            'air_temperature': env_conditions['air_temperature'],
            'humidity': env_conditions['humidity'],
            'cumulative_lap': np.arange(cumulative_lap, cumulative_lap + laps),
            'fuel_load': fuel_load,
            'track_length_km': env_conditions['track_length_km'],
            'number_of_turns': env_conditions['number_of_turns'],
            'average_speed_kmh': env_conditions['average_speed_kmh'],
            'elevation_change_m': env_conditions['elevation_change_m'],
            'tire_compound_Hard': int(compound == 'Hard'),
            'tire_compound_Medium': int(compound == 'Medium'),
            'tire_compound_Soft': int(compound == 'Soft')
        })
        
        # Predict lap times
        predicted_times = model.predict(stint_data[features])
        total_time += predicted_times.sum()
        
        # Add pit stop time except after the last stint
        if idx < len(stints) - 1:
            total_time += pit_duration
        
        cumulative_lap += laps  # Update cumulative lap count
    
    return total_time

# **Step 6: Implement a Genetic Algorithm for Optimization with Tire Compound Constraint**

def genetic_algorithm_optimization(model, env_conditions, population_size=50, generations=20, mutation_rate=0.1):
    # Define the compounds list
    compounds_list = ['Soft', 'Medium', 'Hard']
    
    # Initialize population
    population = []
    for _ in range(population_size):
        # Randomly generate pit laps ensuring they are not the same
        pit_lap1 = random.randint(2, 56)  # Allow for pit_lap2 > pit_lap1
        pit_lap2 = random.randint(pit_lap1 + 1, 57)
        
        # Ensure at least two different tire compounds are used
        while True:
            compound_indices = [random.randint(0, 2) for _ in range(3)]
            compounds_used = set(compound_indices)
            if len(compounds_used) >= 2:
                break  # Valid individual
        individual = [pit_lap1, pit_lap2] + compound_indices
        population.append(individual)
    
    # Evolution process
    for generation in range(generations):
        # Evaluate fitness of each individual
        fitness_scores = []
        for individual in population:
            pit_laps = individual[:2]
            compound_indices = individual[2:]
            compounds_used = set(compound_indices)
            # Check if at least two different tire compounds are used
            if len(compounds_used) < 2:
                total_time = float('inf')  # Assign high penalty
            elif pit_laps[0] >= pit_laps[1]:
                total_time = float('inf')  # Assign high penalty if pit_lap1 >= pit_lap2
            else:
                tire_compounds = [compounds_list[i] for i in compound_indices]
                try:
                    total_time = simulate_race(model, pit_laps, tire_compounds, env_conditions)
                except Exception:
                    total_time = float('inf')  # Assign high penalty if simulation fails
            fitness_scores.append((total_time, individual))
        
        # Sort population by fitness (lower total_time is better)
        fitness_scores.sort(key=lambda x: x[0])
        population = [ind for _, ind in fitness_scores[:population_size//2]]  # Select top 50%
        
        # Generate new population through crossover and mutation
        new_population = population.copy()
        while len(new_population) < population_size:
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            # Crossover
            crossover_point = random.randint(1, 4)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            # Mutation
            if random.random() < mutation_rate:
                mutation_index = random.randint(0, 4)
                if mutation_index < 2:
                    # Mutate pit lap
                    if mutation_index == 0:
                        # Mutate pit_lap1
                        child[0] = random.randint(2, child[1] - 1)
                    else:
                        # Mutate pit_lap2
                        child[1] = random.randint(child[0] + 1, 57)
                else:
                    # Mutate compound index
                    child[mutation_index] = random.randint(0, 2)
            # Enforce constraints after mutation
            # Ensure pit_lap1 < pit_lap2
            if child[0] >= child[1]:
                child[0] = random.randint(2, child[1] - 1)
            # Ensure pit_lap1 >= 2, pit_lap2 <= total_laps - 1
            child[0] = max(2, min(child[0], child[1] - 1))
            child[1] = min(max(child[0] + 1, child[1]), 57)
            # Ensure pit_lap1 != pit_lap2
            if child[0] == child[1]:
                if child[1] < 57:
                    child[1] += 1
                else:
                    child[0] -= 1
            # Enforce the tire compound constraint
            compound_indices = child[2:]
            compounds_used = set(compound_indices)
            if len(compounds_used) < 2:
                # Mutate one of the tire compounds to ensure diversity
                indices_to_change = [i for i in range(2, 5)]
                idx = random.choice(indices_to_change)
                available_compounds = [i for i in range(3) if i not in compounds_used]
                if available_compounds:
                    child[idx] = random.choice(available_compounds)
            new_population.append(child)
        population = new_population
    
    # Get the best individual from the final population
    best_total_time = float('inf')
    best_individual = None
    for ind in population:
        pit_laps = ind[:2]
        compound_indices = ind[2:]
        compounds_used = set(compound_indices)
        if len(compounds_used) < 2:
            continue  # Skip invalid individuals
        if pit_laps[0] >= pit_laps[1]:
            continue  # Skip invalid individuals where pit_lap1 >= pit_lap2
        tire_compounds = [compounds_list[i] for i in compound_indices]
        try:
            total_time = simulate_race(model, pit_laps, tire_compounds, env_conditions)
            if total_time < best_total_time:
                best_total_time = total_time
                best_individual = ind
        except Exception:
            continue
    
    if best_individual is None:
        print("No valid strategy found that meets all constraints.")
        return None
    
    optimal_pit_laps = best_individual[:2]
    optimal_tire_compounds = [compounds_list[i] for i in best_individual[2:]]
    
    # Prepare the strategy
    strategy = {
        'Optimal Pit Laps': optimal_pit_laps,
        'Optimal Tire Compounds': optimal_tire_compounds,
        'Expected Total Race Time (s)': best_total_time
    }
    
    return strategy

# **Step 7: Optimize strategy for the upcoming race using the genetic algorithm**

strategy_new = genetic_algorithm_optimization(best_model, env_conditions_new)

# **Display the optimal strategy**

if strategy_new:
    print("\nOptimal Strategy for Upcoming Race Using Genetic Algorithm with Fuel Load:")
    print(f"Pit Stop Laps: {strategy_new['Optimal Pit Laps']}")
    print(f"Tire Compounds Used: {strategy_new['Optimal Tire Compounds']}")
    print(f"Expected Total Race Time (s): {strategy_new['Expected Total Race Time (s)']:.2f}")
else:
    print("Could not find a valid strategy that meets all constraints.")
