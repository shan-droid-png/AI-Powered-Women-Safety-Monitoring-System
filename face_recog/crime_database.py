import os
from main import add_criminal  # Replace with your script name.

# Directory containing criminal images
image_directory = r"D:\SIH-2024-Project---Women-Safety-USing-CCTV--main (2)\flask\crime_data"

# Adding all criminals to the database
for filename in os.listdir(image_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust file extensions as needed
        name = os.path.splitext(filename)[0]  # Use file name (without extension) as the name
        image_path = os.path.join(image_directory, filename)
        crime_details = f"Details for {name}"  # Replace with actual details if available
        add_criminal(name=name, image_path=image_path, crime_details=crime_details)

print("All criminals added to the database.")
