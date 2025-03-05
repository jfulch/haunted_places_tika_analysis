#!/usr/bin/env python

import os
import csv
import json
import sys

tsv_file_path = "merged_dataset.tsv"
# Output JSON file path
json_file_path = "merged_dataset.json"

# Convert TSV to JSON
def convert_tsv_to_json(tsv_file, json_file):
    data = []
    
    with open(tsv_file, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            # Convert values to appropriate types where possible
            processed_row = {}
            for key, value in row.items():
                if value is None or value.strip() == '':
                    processed_row[key] = None
                elif value.replace('.', '', 1).isdigit():
                    # Try to convert to float or int
                    if '.' in value:
                        processed_row[key] = float(value)
                    else:
                        processed_row[key] = int(value)
                else:
                    processed_row[key] = value
            data.append(processed_row)
    
    # Write to JSON file
    with open(json_file, 'w') as f:
        json.dump({"haunted_places_data": data}, f, indent=2)
    
    print(f"Conversion complete. Converted {len(data)} records.")
    print(f"JSON file saved to: {json_file}")

# Run the conversion
if __name__ == "__main__":
    print(f"Converting TSV file: {tsv_file_path}")
    print(f"Output JSON file: {json_file_path}")
    
    # Check if input file exists
    if not os.path.exists(tsv_file_path):
        print(f"Error: Input TSV file not found at {tsv_file_path}")
        print("Current working directory:", os.getcwd())
        sys.exit(1)
        
    convert_tsv_to_json(tsv_file_path, json_file_path)