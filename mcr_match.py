import os

# Function to read contents of a text file into a list
def read_file_to_list(file_path):
    content_list = []
    with open(file_path, 'r') as file:
        for line in file:
            content_list.append(line.strip())
    return content_list

# Function to check if values match with content lists and append matching file names
def match_and_save(file_path, wt_res_list, position_list, mt_res_list, output_file):
    with open(file_path, 'r') as file:
        for line in file:
            if 'SwapRes' in line:
                line_data = line.split('\t')
                val1 = line_data[0][8:11].strip()
                val2 = line_data[0][12:14].strip()
                val3 = line_data[0][16:19].strip()
                

                # Match with contents of wt_res_list first
                if val1 not in wt_res_list:
                    return  # Exit function if not matched with wt_res_list

                # Then, match with contents of position_list and mt_res_list serially
                if val2 in position_list and val3 in mt_res_list:
                    output_file.write(os.path.basename(file_path) + '\n')  # Save only file name
                    return  # Exit function after first match

# Paths to input files
wt_res_path = '/home/sanjay/Downloads/new_pdbs/wt_res.txt'
position_path = '/home/sanjay/Downloads/new_pdbs/position.txt'
mt_res_path = '/home/sanjay/Downloads/new_pdbs/mt_res.txt'

# Read contents of text files into lists
wt_res_list = read_file_to_list(wt_res_path)
position_list = read_file_to_list(position_path)
mt_res_list = read_file_to_list(mt_res_path)
# Output file to store matching file names
output_file_path = '/home/sanjay/Downloads/new_pdbs/matching_files.txt'

# Iterate over files in the folder for matching
folder_path = '/home/sanjay/Downloads/new_pdbs/folder'
with open(output_file_path, 'w') as output_file:
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            match_and_save(file_path, wt_res_list, position_list, mt_res_list, output_file)
