import os
from Bio.PDB import PDBParser
import plotly.graph_objects as go
import math
import pandas as pd
import json

# Function to calculate hydrophobicity
def calculate_hydrophobicity(residue_name):
    # Hydrophobicity values for amino acid residues
    hydrophobicity_values = {
        'ALA': 1.8, 'CYS': 2.5, 'ASP': -3.5, 'GLU': -3.5, 'PHE': 2.8,
        'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5, 'LYS': -3.9, 'LEU': 3.8,
        'MET': 1.9, 'ASN': -3.5, 'PRO': -1.6, 'GLN': -3.5, 'ARG': -4.5,
        'SER': -0.8, 'THR': -0.7, 'VAL': 4.2, 'TRP': -0.9, 'TYR': -1.3
    }
    return hydrophobicity_values.get(residue_name, 0)

# Function to calculate charge based on pKa values
def calculate_charge(residue_name):
    # Charge values for amino acid residues
    charge_values = {
        'ALA': 0, 'ARG': 1, 'ASN': 0, 'ASP': -1, 'CYS': 0,
        'GLN': 0, 'GLU': -1, 'GLY': 0, 'HIS': 0, 'ILE': 0,
        'LEU': 0, 'LYS': 1, 'MET': 0, 'PHE': 0, 'PRO': 0,
        'SER': 0, 'THR': 0, 'TRP': 0, 'TYR': 0, 'VAL': 0
    }
    return charge_values.get(residue_name, 0)

# Dictionary of molecular weights for each amino acid residue
molecular_weight = {
    'ALA': 89.09, 'CYS': 121.16, 'ASP': 133.10, 'GLU': 147.13, 'PHE': 165.19,
    'GLY': 75.07, 'HIS': 155.16, 'ILE': 131.17, 'LYS': 146.19, 'LEU': 131.17,
    'MET': 149.21, 'ASN': 132.12, 'PRO': 115.13, 'GLN': 146.15, 'ARG': 174.20,
    'SER': 105.09, 'THR': 119.12, 'VAL': 117.15, 'TRP': 204.23, 'TYR': 181.19
}

# Function to calculate distance between two atoms
def calculate_distance(atom1, atom2):
    return math.sqrt(sum((atom1[i] - atom2[i])**2 for i in range(3)))

# Function to check for hydrogen bond between two atoms
def is_hydrogen_bond(donor_atom, hydrogen_atom, acceptor_atom):
    # Check if the distance between donor and acceptor is within hydrogen bond range
    if calculate_distance(donor_atom, acceptor_atom) < 3.5:
        # Check if the angle between donor, hydrogen, and acceptor is within range
        angle = calculate_angle(donor_atom, hydrogen_atom, acceptor_atom)
        if angle > 100 and angle < 180:  # Example angle range for hydrogen bonds
            return True
    return False

# Function to calculate angle between three atoms
def calculate_angle(atom1, atom2, atom3):
    vector1 = [atom1[i] - atom2[i] for i in range(3)]
    vector2 = [atom3[i] - atom2[i] for i in range(3)]
    dot_product = sum(vector1[i] * vector2[i] for i in range(3))
    magnitude1 = math.sqrt(sum(vector1[i] ** 2 for i in range(3)))
    magnitude2 = math.sqrt(sum(vector2[i] ** 2 for i in range(3)))
    angle = math.acos(dot_product / (magnitude1 * magnitude2))
    return math.degrees(angle)

# Function to check for disulfide bond between two sulfur atoms
def is_disulfide_bond(sulfur_atom1, sulfur_atom2):
    # Check if the distance between sulfur atoms is within disulfide bond range
    distance = calculate_distance(sulfur_atom1, sulfur_atom2)
    if distance >= 2.0 and distance <= 2.5:  # Example distance range for disulfide bonds
        return True
    return False

# Function to process PDB file and generate graph data
def process_pdb_file(pdb_file_path):
    # Parse PDB file
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file_path)

    nodes = []  # List to store node information
    edges = []  # List to store edge information
    edge_types = []  # List to store the type of bond for each edge

    # Extract amino acid residues and their features
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == ' ':
                    residue_id = residue.get_id()[1]
                    residue_name = residue.get_resname()
                    # Calculate hydrophobicity
                    hydrophobicity = calculate_hydrophobicity(residue_name)
                    # Calculate charge
                    charge = calculate_charge(residue_name)
                    # Add size of amino acid
                    size = molecular_weight[residue_name]
                    # Extract coordinates of C-alpha atom
                    try:
                        coords = residue['CA'].coord
                        nodes.append({'id': residue_id, 'name': residue_name, 'coords': coords, 'hydrophobicity': hydrophobicity, 'charge': charge, 'size': size})
                    except KeyError:
                        # If C-alpha atom is missing, skip this residue
                        pass

    # Identify interactions (e.g., distance-based) and their types
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i < j:
                distance = calculate_distance(node_i['coords'], node_j['coords'])
                if distance < 10.0:  # Distance cutoff for interaction
                    # Check for hydrogen bonds
                    if is_hydrogen_bond(node_i['coords'], node_j['coords'], node_j['coords']):
                        edges.append((i, j))
                        edge_types.append('hydrogen bond')
                    # Check for disulfide bonds
                    elif is_disulfide_bond(node_i['coords'], node_j['coords']):
                        edges.append((i, j))
                        edge_types.append('disulfide bond')
                    else:
                        # Default to 'covalent' if no specific bond type is identified
                        edges.append((i, j))
                        edge_types.append('covalent')

    return nodes, edges, edge_types

# Directory containing PDB files
pdb_directory = "/home/sanjay/Downloads/AlphaFold_model_PDBs"

# Define the output directory for graph JSON files
output_directory = "/home/sanjay/Downloads/graph/wildgraph"

# Iterate over all PDB files in the directory
for filename in os.listdir(pdb_directory):
    if filename.endswith(".pdb"):
        pdb_file_path = os.path.join(pdb_directory, filename)
        print("Processing:", pdb_file_path)
        
        # Process the PDB file and generate graph data
        nodes, edges, edge_types = process_pdb_file(pdb_file_path)

        # Create a pandas DataFrame from the nodes data
        nodes_df = pd.DataFrame(nodes)

        # Convert the DataFrame to a JSON object
        nodes_json = nodes_df.to_json(orient="records")

        # Similarly, create a DataFrame for edges
        edges_df = pd.DataFrame(edges, columns=["source", "target"])
        edges_df["type"] = edge_types  # Add edge types to the DataFrame

        # Convert the DataFrame to a JSON object
        edges_json = edges_df.to_json(orient="records")

        # Define a dictionary to store the graph information
        graph_data = {
            "nodes": json.loads(nodes_json),
            "edges": json.loads(edges_json)
        }

        # Save graph data to a JSON file in the output directory
        graph_output_file = os.path.join(output_directory, os.path.splitext(filename)[0] + ".json")
        with open(graph_output_file, "w") as json_file:
            json.dump(graph_data, json_file)

        print("Graph data saved to:", graph_output_file)

