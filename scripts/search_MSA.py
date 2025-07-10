import os
import sys
import argparse
import subprocess
from Bio.PDB import PDBParser, PPBuilder
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_seq_from_pdb(pdb_path, output_fasta_path):
    parser = PDBParser(QUIET=True)
    ppb = PPBuilder()
    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]

    try:
        structure = parser.get_structure(pdb_id, pdb_path)
        sequence = ""
        for model in structure:
            for chain in model:
                polypeptides = ppb.build_peptides(chain)
                for poly in polypeptides:
                    sequence += str(poly.get_sequence())
                break
            break

        if sequence:
            record = SeqRecord(Seq(sequence), id=pdb_id, description="")
            SeqIO.write(record, output_fasta_path, "fasta")
        else:
            print(f"Warring: fail get sequence form {pdb_path}")
            return False
        return True
    except Exception as e:
        print(f"Error: can not process {pdb_path},because: {str(e)}")
        return False


def run_msa_search(fasta_file, base_output_dir, flagfile):
    command = [
        "python", "af_multiple_conformation/msas.py",
        "--fasta_paths", fasta_file,
        "--flagfile", flagfile,
        "--output_dir", base_output_dir,
        "--seq_only",
    ]
    try:
        subprocess.run(command, check=True)
        print(f"MSA search : {os.path.basename(fasta_file)} is finish")
    except subprocess.CalledProcessError as e:
        print(f"Error: {fasta_file} ,{e}")

def process_pdb_file(pdb_file, output_fasta_dir, msa_output_dir, flagfile):
    pdb_id = os.path.splitext(os.path.basename(pdb_file))[0]
    fasta_path = os.path.join(output_fasta_dir, f"{pdb_id}.fasta")
    os.makedirs(output_fasta_dir, exist_ok=True)

    if extract_seq_from_pdb(pdb_file, fasta_path):
        run_msa_search(fasta_path, msa_output_dir, flagfile)

def main(pdb_input_dir, fasta_output_dir, msa_output_dir, flagfile, num_threads):
    pdb_files = [os.path.join(pdb_input_dir, f) for f in os.listdir(pdb_input_dir) if f.endswith(".pdb")]
    os.makedirs(msa_output_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for pdb_file in pdb_files:
            futures.append(executor.submit(
                process_pdb_file, pdb_file, fasta_output_dir, msa_output_dir, flagfile
            ))

        for future in as_completed(futures):
            pass  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSA search")
    parser.add_argument("--pdb_dir", type=str, required=True, help="pdb_dir")
    parser.add_argument("--fasta_dir", type=str, required=True, help="fasta_dir")
    parser.add_argument("--msa_out_dir", type=str, required=True, help="msa_out_dir")
    parser.add_argument("--flagfile", type=str, required=True, help="flagfile")
    parser.add_argument("--num_threads", type=int, default=1, help="num_threads")
    args = parser.parse_args()

    main(args.pdb_dir, args.fasta_dir, args.msa_out_dir, args.flagfile, args.num_threads)
