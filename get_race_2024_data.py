import requests
import pandas as pd


def get_session_keys(grand_prix, year=2024):
    url = "https://api.openf1.org/v1/sessions"
    params = {
        'country_name': grand_prix,
        'year': year
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        # Filter sessions for Qualifying and Race
        race_sessions = [session for session in data if session['session_name'] in [
            'Qualifying', 'Race']]
        return {session['session_name']: (session['session_key'], session['circuit_short_name']) for session in race_sessions}
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
    'Monte Carlo': 3.318, 'Shanghai': 5.451, 'Suzuka': 5.807, 'Melbourne': 5.029, 'Jeddah': 6.174, 'Sakhir': 5.412
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


def collect_data_for_race():
    grand_prix = ["Singapore", "Azerbaijan", "Italy", "Netherlands", "Belgium", "Hungary", "Great Britain", "Austria", "Spain",
                  "Canada", "Monaco", "Emilia-Romagna", "Miami", "China", "Japan", "Australia", "Saudi Arabia", "Bahrain"]

    all_gp_data = {}

    for gp in grand_prix:
        session_keys = get_session_keys(gp)

        if not session_keys:
            continue

        all_data = []

        for session_name, (session_key, circuit_short_name) in session_keys.items():
            print(
                f"Collecting data for {gp} - {session_name} (Session key: {session_key})...")

            lap_data = get_lap_data(session_key)
            weather_data = get_weather_data(session_key)
            stints_data = get_stints_data(session_key)
            driver_data = get_driver_data(session_key)

            print(
                f"Driver data for {session_name} (Session key: {session_key}):")
            print(driver_data)

            mclaren_drivers = [driver['driver_number']
                               for driver in driver_data if driver['team_name'] == "McLaren"]

            print(f"McLaren drivers for {session_name}: {mclaren_drivers}")

            if session_name == 'Race' and not mclaren_drivers:
                mclaren_drivers = [4, 81]
                print(
                    f"Falling back to known McLaren drivers for race: {mclaren_drivers}")

            if lap_data and weather_data and stints_data:
                for lap in lap_data:
                    if lap['driver_number'] in mclaren_drivers:
                        stint = next(
                            (s for s in stints_data 
                             if s['driver_number'] == lap['driver_number'] and 
                             s['lap_start'] <= lap['lap_number'] <= s['lap_end']), 
                            None)
                        #if stint and stint['compound'] != 'MEDIUM':
                        #    print(f"Compound changed to {stint['compound']} for driver {lap['driver_number']} on lap {lap['lap_number']}")
                        
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

        all_gp_data[gp] = pd.DataFrame(all_data)

        for g_p, data in all_gp_data.items():
            if not data.empty:
                all_gp_data[g_p] = calculate_distance_and_speed(data)
            else:
                print(f"No data available for {g_p}")

    with pd.ExcelWriter('mclaren_races_2024_data.xlsx') as writer:
        for gp, data in all_gp_data.items():
            data.to_excel(writer, sheet_name=gp, index=False)

    print("Data saved to mclaren_races_2024_data.xlsx")
    print()


collect_data_for_race()
