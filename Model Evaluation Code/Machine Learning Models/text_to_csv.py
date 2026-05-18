import csv

# Input and output file paths
input_file = 'merged.txt'   # Replace with your actual input file name
output_file = 'our_dataset.csv'

# Open the input text file and output CSV file
with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    
    csv_writer = csv.writer(outfile)
    # Write header row
    csv_writer.writerow(['query', 'label'])

    # Process each line
    for line in infile:
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        try:
            query, label = line.split('::::')
            csv_writer.writerow([query.strip(), label.strip()])
        except ValueError:
            print(f"Skipping malformed line: {line}")
