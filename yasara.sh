
#!/bin/bash

# Directory containing the text files
folder="/home/sanjay/Desktop/sanjay"

# Get a list of all text files in the folder
files=("$folder"/*.mcr)

# Initialize a counter for the number of iterations
count=0

# Loop through each file and execute Yasara
for file in "${files[@]}"; do
    ((count++))
    echo "Iteration $count: Executing file: $file"
    /home/sanjay/yasara/yasara -txt "$file"
done

echo "Total iterations: $count"

