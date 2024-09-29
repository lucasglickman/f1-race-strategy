import requests
import pandas as pd


def get_session_keys(grand_prix, shortname, year=2023):
    url = "https://api.openf1.org/v1/sessions"
    params = {
        'year': year
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        # Filter sessions for Practice 1, 2, and 3 and match with grand_prix
        practice_sessions = [
            session for session in data 
            if session['session_name'] in ['Practice 1', 'Practice 2', 'Practice 3'] 
            and (session['country_name'] == grand_prix and session['circuit_short_name'] == shortname)
        ]
        return {session['session_name']: (session['session_key'], session['circuit_short_name']) for session in practice_sessions}
    else:
        print(
            f"Failed to retrieve sessions. Status code: {response.status_code}")
        return None


def get_lap_data(session_key):
    url = "https://api.openf1.org/v1/laps"
    params = {
        'session_key': session_key
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(
            f"Failed to retrieve lap data. Status code: {response.status_code}")
        return None


def get_weather_data(session_key):
    url = "https://api.openf1.org/v1/weather"
    params = {
        'session_key': session_key
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(
            f"Failed to retrieve weather data. Status code: {response.status_code}")
        return None


def get_stints_data(session_key):
    url = "https://api.openf1.org/v1/stints"
    params = {
        'session_key': session_key
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(
            f"Failed to retrieve stints data. Status code: {response.status_code}")
        return None


def get_driver_data(session_key):
    url = "https://api.openf1.org/v1/drivers"
    params = {
        'session_key': session_key
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(
            f"Failed to retrieve driver data. Status code: {response.status_code}")
        return None


track_lengths = {
    'Singapore': 5.063, 'Baku': 6.003, 'Monza': 5.793, 'Zandvoort': 4.259, 'Spa-Francorchamps': 7.004,
    'Hungaroring': 4.381, 'Silverstone': 5.891, 'Spielberg': 4.326, 'Catalunya': 4.655, 'Montreal': 4.421,
    'Monte Carlo': 3.318, 'Shanghai': 5.451, 'Suzuka': 5.807, 'Melbourne': 5.029, 'Jeddah': 6.174, 'Sakhir': 5.412,
    'Austin': 5.514, 'Mexico City': 4.304, 'Miami': 5.412, 'Interlagos': 4.309, 'Las Vegas': 6.201, 'Losail': 5.380, 
    'Yas Marina Circuit': 5.554
}


def calculate_distance_and_speed(df):
    df['distance_traveled'] = df['circuit_short_name'].map(
        track_lengths) * (df['lap_number'] + df['tire_age_at_start'])
    df['avg_speed'] = df['circuit_short_name'].map(
        track_lengths) / df['lap_time'] * 1000

    missing_lengths = df['circuit_short_name'].map(track_lengths).isnull()

    if missing_lengths.any():
        print(
            f"Missing track lengths for: {df['circuit_short_name'][missing_lengths].unique()}")

    return df


def collect_data_for_practice(grand_prix, shortname):
    session_keys = get_session_keys(grand_prix, shortname)

    if not session_keys:
        print(f"No session keys found for {grand_prix}")
        return

    all_data = []

    for session_name, (session_key, circuit_short_name) in session_keys.items():
        print(
            f"Collecting data for {grand_prix} - {session_name} (Session key: {session_key})...")

        lap_data = get_lap_data(session_key)
        weather_data = get_weather_data(session_key)
        stints_data = get_stints_data(session_key)
        driver_data = get_driver_data(session_key)

        if lap_data and weather_data and stints_data:
            for lap in lap_data:
                if lap['driver_number'] in [4, 81]:  # Filter for driver numbers 4 and 81
                    stint = next(
                        (s for s in stints_data 
                         if s['driver_number'] == lap['driver_number'] and 
                         s['lap_start'] <= lap['lap_number'] <= s['lap_end']), 
                        None)
                    
                    weather = weather_data[0] if weather_data else {}

                    all_data.append({
                        'session': session_name,
                        'driver_number': lap['driver_number'],
                        'lap_number': lap['lap_number'],
                        'lap_time': lap['lap_duration'],
                        'sector_1_time': lap['duration_sector_1'],
                        'sector_2_time': lap['duration_sector_2'],
                        'sector_3_time': lap['duration_sector_3'],
                        'tire_compound': stint['compound'] if stint else None,
                        'tire_age_at_start': stint['tyre_age_at_start'] if stint else None,
                        'track_temperature': weather.get('track_temperature'),
                        'air_temperature': weather.get('air_temperature'),
                        'wind_speed': weather.get('wind_speed'),
                        'humidity': weather.get('humidity'),
                        'air_pressure': weather.get('pressure'),
                        'circuit_short_name': circuit_short_name
                    })

    df = pd.DataFrame(all_data)
    
    if not df.empty:
        df = calculate_distance_and_speed(df)
        
        excel_filename = 'new_practice_data.xlsx'
        with pd.ExcelWriter(excel_filename) as writer:
            df.to_excel(writer, sheet_name=grand_prix, index=False)
        
        print(f"Data saved to {excel_filename}")
    else:
        print(f"No data available for {grand_prix}")


#Insert grand prix name here (country name)

"""
Australia
Austria
Azerbaijan
Bahrain
Belgium
Brazil
Canada
Great Britain
Hungary
Italy
Japan
Mexico
Monaco
Netherlands
Qatar
Saudi Arabia
Singapore
Spain
United Arab Emirates
United States
"""


grand_prix = "United States"
shortname = "Miami"

collect_data_for_practice(grand_prix, shortname)
