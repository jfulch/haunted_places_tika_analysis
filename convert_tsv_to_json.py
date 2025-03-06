#!/usr/bin/env python

import os
import sys
import json
import etl.tsvtojson  # This imports the ETLLib tsv2json tool

# File paths
tsv_file_path = "datasets/merged_dataset.tsv"
json_file_path = "datasets/merged_dataset.json"
column_headers_path = "datasets/headers.txt"

def create_headers_file(tsv_file, headers_file):
    """Create a headers file from the first line of the TSV"""
    if os.path.exists(headers_file):
        print(f"Headers file already exists at {headers_file}")
        return
    
    try:
        with open(tsv_file, 'r', encoding='utf-8') as f:
            header_line = f.readline().strip()
        
        with open(headers_file, 'w', encoding='utf-8') as f:
            f.write(header_line)
        
        print(f"Created headers file at {headers_file}")
    except Exception as e:
        print(f"Error creating headers file: {str(e)}")
        sys.exit(1)

def run_tsv2json(tsv_file, json_file, headers_file):
    """Use ETLLib's tsv2json tool to convert TSV to JSON"""
    print(f"Converting TSV to JSON using ETLLib's tsv2json tool")
    print(f"Input: {tsv_file}")
    print(f"Output: {json_file}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    
    # Save original argv
    original_argv = sys.argv.copy()
    
    try:
        # Configure the tool with proper arguments
        # Note: Using verbose mode to see what's happening
        sys.argv = [
            'tsvtojson.py',
            '-v',                  # Verbose mode
            '-t', tsv_file,        # Input TSV file
            '-j', json_file,       # Output JSON file
            '-c', headers_file,    # Column headers file
            '-o', 'haunted_places' # Object type
        ]
        
        # Run the ETLLib tool
        etl.tsvtojson.main()
        
        # Restore original argv
        sys.argv = original_argv
        
        if os.path.exists(json_file) and os.path.getsize(json_file) > 0:
            print(f"Conversion successful! JSON file created at {json_file}")
            return True
        else:
            print(f"Warning: JSON file not created or empty")
            return False
            
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False
    
def convert_tsv_directly(tsv_file, json_file):
    """Convert TSV to JSON directly with proper field mapping and character handling"""
    print(f"Directly converting TSV to well-formatted JSON")
    print(f"Input: {tsv_file}")
    print(f"Output: {json_file}")
    
    try:
        # Read the TSV file
        with open(tsv_file, 'r', encoding='utf-8') as f:
            # Get headers from first line
            headers = f.readline().strip().split('\t')
            
            # Clean headers - remove problematic characters
            clean_headers = [header.strip().replace('\t', ' ').replace('\\', '') for header in headers]
            
            # Process data rows
            data = []
            line_count = 0
            for line in f:
                line_count += 1
                try:
                    values = line.strip().split('\t')
                    
                    # Create record with proper field names and clean values
                    record = {}
                    # Add all fields, even if empty
                    for i, header in enumerate(clean_headers):
                        if i < len(values):
                            clean_value = values[i].strip().replace('\t', ' ')
                            record[header] = clean_value if clean_value else ""
                        else:
                            record[header] = ""
                    
                    # Only add non-empty records
                    if record:
                        data.append(record)
                        
                except Exception as e:
                    print(f"Warning: Could not process line {line_count}: {str(e)}")
                    continue
        
        # Create well-structured JSON
        output = {"haunted_places": data}
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        
        # Write to file with nice formatting using ensure_ascii=False for better Unicode handling
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully converted {len(data)} records to JSON")
        return True
    except Exception as e:
        print(f"Error during direct conversion: {str(e)}")
        return False
    
def convert_tsv_to_json(input_tsv, output_json):

    print(f"Converting TSV file: {input_tsv}")
    print(f"Output JSON file: {output_json}")
    
    # Create headers file
    headers_file = os.path.join(os.path.dirname(output_json), "headers.txt")
    create_headers_file(input_tsv, headers_file)
    
    # Try ETLLib approach first
    etllib_success = run_tsv2json(input_tsv, output_json, headers_file)
    
    # If ETLLib approach fails, use direct conversion
    if not etllib_success or not os.path.exists(output_json) or os.path.getsize(output_json) < 10:
        print("ETLLib conversion didn't produce usable output, using direct conversion...")
        convert_tsv_directly(input_tsv, output_json)
    
    # Verify the result
    if os.path.exists(output_json) and os.path.getsize(output_json) > 0:
        print(f"Conversion successful! JSON file created at {output_json}")
        return True
    else:
        print(f"Conversion failed! Could not create JSON file at {output_json}")
        return False

# Main execution
if __name__ == "__main__":
    # Step 1: Create headers file
    create_headers_file(tsv_file_path, column_headers_path)
    
    # Step 2: Try ETLLib's tsv2json first (as required by assignment)
    print("\n=== Attempting conversion with ETLLib ===")
    etllib_success = run_tsv2json(tsv_file_path, json_file_path, column_headers_path)
    
    # Step 3: If ETLLib's tool didn't produce good output, use direct conversion
    if not etllib_success or not os.path.exists(json_file_path) or os.path.getsize(json_file_path) < 10:
        print("\n=== ETLLib conversion didn't produce usable output ===")
        print("=== Using direct conversion method instead ===")
        direct_json_path = json_file_path.replace('.json', '_direct.json')
        convert_tsv_directly(tsv_file_path, direct_json_path)
        json_file_path = direct_json_path  # Use the direct conversion output
    
    # Step 4: Verify and display results
    if os.path.exists(json_file_path):
        print("\n=== Conversion completed successfully! ===")
        print(f"Output JSON file: {json_file_path}")
        
        # Display sample of the output
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                record_count = len(data.get("haunted_places", []))
                print(f"Found {record_count} records in the JSON output")
                
                # Show sample of first record
                if record_count > 0:
                    first_record = data["haunted_places"][0]
                    print("\nSample record (first entry):")
                    sample_json = json.dumps(first_record, indent=2, ensure_ascii=False)
                    print(sample_json[:500] + "..." if len(sample_json) > 500 else sample_json)
        except Exception as e:
            print(f"Warning: Error reading the JSON output file: {str(e)}")
    else:
        print("Conversion failed - no output file was created.")
    
    # Step 4: Verify and display results
    if os.path.exists(json_file_path):
        print("\n=== Conversion completed successfully! ===")
        print(f"Output JSON file: {json_file_path}")
        
        # Display sample of the output
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                record_count = len(data.get("haunted_places", []))
                print(f"Found {record_count} records in the JSON output")
                
                # Show sample of first record
                if record_count > 0:
                    first_record = data["haunted_places"][0]
                    print("\nSample record (first entry):")
                    sample_json = json.dumps(first_record, indent=2, ensure_ascii=False)
                    print(sample_json[:500] + "..." if len(sample_json) > 500 else sample_json)
        except Exception as e:
            print(f"Warning: Error reading the JSON output file: {str(e)}")
    else:
        print("Conversion failed - no output file was created.")