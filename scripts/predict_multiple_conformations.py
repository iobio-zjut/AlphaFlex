#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def process_file(a3m_file_path, protein_folder_name, output_base_dir, af2_root_dir):
    """
    Process a single .a3m file by calling RunAF2.py.

    Args:
        a3m_file_path (str): Path to the .a3m input file.
        protein_folder_name (str): Name of the parent protein folder.
        output_base_dir (str): Base output directory.
        af2_root_dir (str): Path to AlphaFold2 installation.
    """
    a3m_file = Path(a3m_file_path)
    output_pdb_subdir = Path(output_base_dir) / protein_folder_name / "pdb"
    output_pdb_subdir.mkdir(parents=True, exist_ok=True)

    pdb_filename = a3m_file.with_suffix(".pdb").name
    final_pdb_path = output_pdb_subdir / pdb_filename

    if final_pdb_path.exists():
        log(f"[SKIP] PDB already exists: {final_pdb_path.name}")
        return True

    log(f"[START] {a3m_file.name} from {protein_folder_name}")

    command = [
        "python",
        "./scripts/RunAF2.py",
        str(a3m_file),
        "--af2_dir", af2_root_dir,
        "--output_dir", str(output_pdb_subdir),
        "--model_num", "1",
        "--recycles", "3",
        "--seed", "0"
    ]

    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True
        )
        log(f"[DONE] {a3m_file.name}")
        return True

    except subprocess.CalledProcessError as e:
        log(f"[ERROR] Failed: {a3m_file.name}, Exit {e.returncode}")
        log(f"STDERR: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch run AlphaFold2 on A3M files using RunAF2.py")
    parser.add_argument("--input_base_dir", required=True,
                        help="Base input directory containing protein folders.")
    parser.add_argument("--output_base_dir", required=True,
                        help="Base output directory (usually same as input).")
    parser.add_argument("--folders_file", required=True,
                        help="Text file listing protein folder names to process.")
    parser.add_argument("--num_threads", type=int, default=6,
                        help="Maximum number of concurrent threads.")
    parser.add_argument("--af2_dir", required=True,
                        help="Path to AlphaFold2 installation.")

    args = parser.parse_args()

    log("Job started.")

    # Load protein folder list
    if not os.path.isfile(args.folders_file):
        log(f"Error: Cannot find folder list file: {args.folders_file}")
        sys.exit(1)

    with open(args.folders_file) as f:
        folder_list = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not folder_list:
        log("Error: No valid folder names found.")
        sys.exit(1)

    tasks = []

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        for protein_folder in folder_list:
            input_dir = Path(args.input_base_dir) / protein_folder / "sub_msas"
            if not input_dir.is_dir():
                log(f"[WARN] Missing folder: {input_dir}")
                continue

            a3m_files = list(input_dir.glob("*.a3m"))
            if not a3m_files:
                log(f"[SKIP] No .a3m files in {input_dir}")
                continue

            log(f"[FOLDER] {protein_folder} — {len(a3m_files)} .a3m files")
            for a3m_path in a3m_files:
                future = executor.submit(
                    process_file,
                    str(a3m_path),
                    protein_folder,
                    args.output_base_dir,
                    args.af2_dir
                )
                tasks.append(future)

        # Wait for all tasks to complete
        for future in as_completed(tasks):
            future.result()

    log("Job finished.")


if __name__ == "__main__":
    main()
