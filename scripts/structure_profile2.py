import os
import subprocess
import numpy as np
import argparse
from Bio import PDB
from Bio.PDB.Polypeptide import is_aa
from concurrent.futures import ThreadPoolExecutor, as_completed


TMalign = "TMalign"

AA_DICT = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}

def three_to_one(residue_name):
    return AA_DICT.get(residue_name, "X")

def extract_sequences_from_pdb(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_file), pdb_file)
    sequences = []
    for model in structure:
        for chain in model:
            seq = ''.join(three_to_one(res.resname) for res in chain if is_aa(res, standard=True))
            if seq:
                sequences.append((chain.id, seq))
    return sequences

def extract_residues_from_pdb(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_file), pdb_file)
    return [res for model in structure for chain in model for res in chain if is_aa(res, standard=True)]

def get_ca_coordinate(residue):
    if "CA" in residue:
        return residue["CA"].get_coord()
    elif "C" in residue:
        return residue["C"].get_coord()
    return None

def process_subfolder(query_name, args):
    query_pdb_path = os.path.join(args.query_dir, query_name, f"{query_name}.pdb")
    if not os.path.exists(query_pdb_path):
        print(f"Skipping {query_name}: No matching PDB file found.")
        return

    target_subdir = os.path.join(args.target_dir, query_name)
    if not os.path.exists(target_subdir):
        print(f"Skipping {query_name}: target subfolder not found.")
        return

    align_subdir = os.path.join(args.align_dir, query_name)
    os.makedirs(align_subdir, exist_ok=True)

    output_file = os.path.join(align_subdir, f"{query_name}.txt")
    with open(output_file, "w") as f:
        f.write(f"{query_name}\n")

    query_sequences = extract_sequences_from_pdb(query_pdb_path)
    query_sequence = query_sequences[0][1] if query_sequences else ''
    query_sequence_written = False
    query_residues = extract_residues_from_pdb(query_pdb_path)

    for target_pdb in os.listdir(target_subdir):
        if not target_pdb.endswith(".pdb"):
            continue
        target_pdb_path = os.path.join(target_subdir, target_pdb)
        target_pdb_id = target_pdb[:-4]

        cmd = [os.path.join(args.tmalign_path, "TMalign"), query_pdb_path, target_pdb_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        lines = result.stdout.split("\n")

        sequences, alignment_line = [], ""
        for line in lines:
            if line and not any(c in line for c in ":. "):
                sequences.append(line.strip())
            if ":" in line or "." in line:
                alignment_line = line.strip()

        if len(sequences) != 2:
            continue

        aligned_target, aligned_template = sequences
        matching_residues = []
        target_counter = template_counter = 0

        for i in range(len(alignment_line)):
            t_res, tmpl_res, sym = aligned_target[i], aligned_template[i], alignment_line[i]
            if sym in {':', '.'} and t_res != '-' and tmpl_res != '-':
                matching_residues.append({
                    'target_residue': t_res, 'target_pos': target_counter + 1,
                    'template_residue': tmpl_res, 'template_pos': template_counter + 1,
                    'align_symbol': sym
                })
            if t_res != '-': target_counter += 1
            if tmpl_res != '-': template_counter += 1

        target_structure_residues = extract_residues_from_pdb(target_pdb_path)
        L = len(query_sequence)
        result_tensor = np.full((L, 1), -1.0, dtype=float)

        for match in matching_residues:
            q_idx = match['target_pos'] - 1
            t_idx = match['template_pos'] - 1
            if q_idx < len(query_residues) and t_idx < len(target_structure_residues):
                coord1 = get_ca_coordinate(query_residues[q_idx])
                coord2 = get_ca_coordinate(target_structure_residues[t_idx])
                distance = np.linalg.norm(coord1 - coord2) if coord1 is not None and coord2 is not None else -1.0
                result_tensor[q_idx, 0] = distance

        with open(output_file, "a") as f:
            if not query_sequence_written:
                f.write(f"{query_sequence}\n")
                query_sequence_written = True
            f.write(f"{target_pdb_id}\n")
            for match in matching_residues:
                f.write(f"Target {match['target_residue']} (Pos {match['target_pos']}) <-> "
                        f"Template {match['template_residue']} (Pos {match['template_pos']}), "
                        f"Symbol: {match['align_symbol']}\n")

        np.savez(os.path.join(align_subdir, f"{target_pdb_id}.npz"), data=result_tensor)

def main():
    parser = argparse.ArgumentParser(description="Run TM-align-based structure alignment.")
    parser.add_argument("--tmalign_path", required=True, help="Path to TM-align binary directory")
    parser.add_argument("--query_dir", required=True, help="Directory containing query subfolders with PDBs")
    parser.add_argument("--target_dir", required=True, help="Directory containing foldseek search results")
    parser.add_argument("--align_dir", required=True, help="Output directory for alignment results")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of threads")
    parser.add_argument("--filter_list", default=None, help="Optional: file listing query names to process")
    args = parser.parse_args()

    os.makedirs(args.align_dir, exist_ok=True)
    os.makedirs(args.target_dir, exist_ok=True)

    all_subfolders = [d for d in os.listdir(args.query_dir) if os.path.isdir(os.path.join(args.query_dir, d))]
    if args.filter_list and os.path.isfile(args.filter_list):
        with open(args.filter_list, 'r') as f:
            filtered = set(line.strip() for line in f if line.strip())
        subfolders = [d for d in all_subfolders if d in filtered]
    else:
        subfolders = all_subfolders

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_subfolder, query_name, args) for query_name in subfolders]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing a subfolder: {e}")

if __name__ == "__main__":
    main()
