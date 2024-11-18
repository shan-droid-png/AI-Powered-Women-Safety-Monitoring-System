import requests
import folium
import time
import os

# Function to get live coordinates using an IP-based geolocation API
def get_live_coordinates():
    try:
        # Using a free API for geolocation (ip-api.com)
        response = requests.get("http://ip-api.com/json/")
        data = response.json()

        if data['status'] == 'success':
            latitude = data['lat']
            longitude = data['lon']
            print(f"Current Coordinates: Latitude={latitude}, Longitude={longitude}")
            return latitude, longitude
        else:
            print(f"Error fetching location: {data.get('message', 'Unknown error')}")
            return None, None

    except requests.exceptions.RequestException as e:
        print(f"Network error occurred while fetching location: {e}")
        return None, None
    except Exception as e:
        print(f"Error occurred while fetching live location: {e}")
        return None, None

# Function to create an interactive map with radar effect
def create_map_with_radar(latitude, longitude):
    """
    Creates a folium map centered around the given coordinates with a radar circle effect.
    """
    map_location = folium.Map(location=[latitude, longitude], zoom_start=14)

    # Add a red radar effect using a circle
    folium.Circle(
        location=[latitude, longitude],
        radius=100,  # Radius in meters
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.3,
    ).add_to(map_location)

    # Add a marker at the center with coordinates displayed in the popup
    folium.Marker([latitude, longitude], popup=f"Lat: {latitude}, Lon: {longitude}").add_to(map_location)

    return map_location

def save_map(map_location, file_path):
    """
    Saves the map as an HTML file to the specified location.
    """
    try:
        map_location.save(file_path)
        print(f"Map saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving map: {e}")

def main():
    # Define the location of the map file
    map_file_path = "C:\\Users\\user\\Desktop\\SIH 2024\\GIS\\templates\\location_map.html"

    # Fetch initial location
    print("Fetching initial location...")
    latitude, longitude = get_live_coordinates()
    if latitude is None or longitude is None:
        print("Unable to fetch initial location. Exiting.")
        return  # Exit the program if initial location couldn't be fetched

    # Create initial map with radar and save it
    map_location = create_map_with_radar(latitude, longitude)
    save_map(map_location, map_file_path)

    # Periodically update location
    while True:
        print("Fetching updated location...")
        latitude, longitude = get_live_coordinates()

        if latitude is not None and longitude is not None:
            print(f"Updated location: Lat: {latitude}, Lon: {longitude}")

            # Create a new map with updated location and radar effect
            map_location = create_map_with_radar(latitude, longitude)

            # Save the updated map to the file
            save_map(map_location, map_file_path)
        else:
            print("Failed to fetch updated location. Retrying...")

        # Sleep for 100 seconds before the next update
        time.sleep(100)

if __name__ == "_main_":
    main()