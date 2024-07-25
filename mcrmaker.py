# Read mt_res.txt into a list
with open('/home/sanjay/Desktop/mutant.txt', 'r') as mt_res_file:
    mut_res = mt_res_file.readlines()

# Read position.txt into a list
with open('/home/sanjay/Desktop/pos.txt', 'r') as position_file:
    position = position_file.readlines()

# Read wt_res.txt into a list
with open('/home/sanjay/Desktop/wild_type.txt', 'r') as wt_res_file:
    wt_res = wt_res_file.readlines()

# Read pdbname_list_with_path.txt into a list
with open('/home/sanjay/Desktop/newoutput.txt', 'r') as pdb_file:
    pdb_files = pdb_file.readlines()

# Read new_pdb_names.txt into a list
with open('/home/sanjay/Desktop/original_pdbnames.txt', 'r') as new_pdb_name_file:
    new_pdb_name = new_pdb_name_file.readlines()

# Strip newline characters from each element in the lists
mut_res = [line.strip() for line in mut_res]
position = [line.strip() for line in position]
wt_res = [line.strip() for line in wt_res]
pdb_files = [line.strip() for line in pdb_files]
new_pdb_name = [line.strip() for line in new_pdb_name]

for mut, pos, wt, pdb, new_name in zip(mut_res, position, wt_res, pdb_files, new_pdb_name):
    
    output_filename = f"/home/sanjay/Desktop/sanjay/{new_name}.mcr"  # Name each output file uniquely
    with open(output_filename, 'w') as output_file:
        output_file.write(f"LoadPDB {pdb}\n")
        output_file.write(f"SwapRes {wt} {pos},{mut},Isomer=L\n")
        output_file.write(f"SavePDB 1,/home/sanjay/Desktop/sanjay/new_pdbs/{new_name}.pdb,Format=PDB,Transform=Yes\n")
        output_file.write("exit")
