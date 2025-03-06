#!/usr/bin/env python

import json
import os
import re

def break_json(json_file="merged_dataset.json", output_dir="haunted_places_files"):
    """Break JSON file into individual text files.
    
    Args:
        json_file: Path to the JSON file
        output_dir: Directory to store individual files
    """
    # Create output directory for individual files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Process each record
    for i, place in enumerate(data["haunted_places"]):
        # Create a clean filename from location name or use index if not available
        if "location" in place and place["location"]:
            filename = re.sub(r'[^\w\s-]', '', place["location"])
            filename = re.sub(r'[-\s]+', '_', filename)
        else:
            filename = f"place_{i}"
        
        # Limit filename length
        filename = filename[:50]
        
        # Create a text file with place details
        with open(f"{output_dir}/{filename}.txt", 'w') as f:
            for key, value in place.items():
                if value is not None:
                    f.write(f"{key}: {value}\n")
    
    print(f"Created {len(data['haunted_places'])} individual files in {output_dir}/")

if __name__ == "__main__":
    break_json()