import json

def preprocess_hospital_data(input_file, output_file):
    """
    Preprocess hospital data according to specified requirements:
    - Convert MinFees/MaxFees to "Fees" format (min-max)
    - Fill missing values with "-"
    - Remove duplicates
    - Remove records missing Name or Specialization
    - Combine Phone and email if in separate attributes
    """
    
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    seen_records = set()  # To track duplicates
    
    for record in data:
        # Skip records without Name or Specialization
        if not record.get('Name') or not record.get('Specialization'):
            continue
        
        # Create a new clean record
        clean_record = {}
        
        # Standard fields with default "-" for missing or empty values
        clean_record['Name'] = record.get('Name', '-').strip() if record.get('Name', '').strip() else '-'
        clean_record['City'] = record.get('City', '-').strip() if record.get('City', '').strip() else '-'
        clean_record['Province'] = record.get('Province', '-').strip() if record.get('Province', '').strip() else '-'
        clean_record['Website'] = record.get('Website', '-').strip() if record.get('Website', '').strip() else '-'
        clean_record['Address'] = record.get('Address', '-').strip() if record.get('Address', '').strip() else '-'
        clean_record['Specialization'] = record.get('Specialization', '-').strip() if record.get('Specialization', '').strip() else '-'
        clean_record['ContactPerson'] = record.get('ContactPerson', '-').strip() if record.get('ContactPerson', '').strip() else '-'
        
        # Handle Phone and email combination
        phone = record.get('Phone', '').strip()
        
        # Check if phone already contains email (has @ symbol)
        if '@' in phone:
            clean_record['Phone'] = phone if phone else '-'
        elif phone:
            # Phone exists, check for separate email
            email = record.get('email', '').strip()
            if email:
                clean_record['Phone'] = f"{phone} {email}"
            else:
                clean_record['Phone'] = phone
        else:
            clean_record['Phone'] = '-'
        
        # Handle Fees - convert MinFees/MaxFees to "min-max" format or single value
        if 'Fees' in record and record['Fees']:
            # Already in correct format, just use it
            fees_str = str(record['Fees']).strip()
            if fees_str and fees_str != '-':
                clean_record['Fees'] = fees_str
            else:
                clean_record['Fees'] = '-'
        elif 'MinFees' in record and 'MaxFees' in record:
            # Both available - use "min-max" format
            min_fee = record['MinFees']
            max_fee = record['MaxFees']
            clean_record['Fees'] = f"{min_fee}-{max_fee}"
        elif 'MinFees' in record:
            # Only MinFees available - use single value
            clean_record['Fees'] = str(record['MinFees'])
        elif 'MaxFees' in record:
            # Only MaxFees available - use single value
            clean_record['Fees'] = str(record['MaxFees'])
        else:
            # No fee information
            clean_record['Fees'] = '-'
        
        # Create a unique identifier for duplicate detection
        # Using Name, City, and Address as unique combination
        unique_key = f"{clean_record['Name']}|{clean_record['City']}|{clean_record['Address']}"
        
        # Skip duplicates
        if unique_key in seen_records:
            continue
        
        seen_records.add(unique_key)
        processed_data.append(clean_record)
    
    # Write the processed data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    
    print(f"Processing complete!")
    print(f"Original records: {len(data)}")
    print(f"Processed records: {len(processed_data)}")
    print(f"Records removed: {len(data) - len(processed_data)}")
    print(f"Output saved to: {output_file}")

# Usage
if __name__ == "__main__":
    input_file = "allHospitals.json"  # Your input file
    output_file = "hospitals.json"  # Output file
    
    preprocess_hospital_data(input_file, output_file)