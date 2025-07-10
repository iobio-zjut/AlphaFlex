import os
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

def find_and_copy_pdb(pdb_id, root, template_dir, output_dir):
    pdb_file_path = os.path.join(template_dir, f'{pdb_id}.pdb')
    if os.path.exists(pdb_file_path):
        output_pdb_dir = os.path.join(output_dir, os.path.relpath(root, args.input_dir))
        os.makedirs(output_pdb_dir, exist_ok=True)
        shutil.copy(pdb_file_path, os.path.join(output_pdb_dir, f'{pdb_id}.pdb'))
        return None
    else:
        return f'Not found: {pdb_id}.pdb in template library.'


def process_m8_file(m8_file_path, root, template_dir, output_dir):
    results = []
    with open(m8_file_path, 'r') as m8_file:
        lines = [line.strip() for line in m8_file if line.strip()]
        if not lines:
            return ['Empty m8 file: ' + m8_file_path]

        target_id = lines[0].split()[0]
        query_ids = set()

        for line in lines:
            parts = line.split()
            if len(parts) > 1:
                query_id = parts[1]
                if query_id != target_id:
                    query_ids.add(query_id)

        if not query_ids:
            return [f'No valid queries found for {target_id}']

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(find_and_copy_pdb, qid, root, template_dir, output_dir) for qid in query_ids]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

    return results

def process_directory(input_dir, template_dir, output_dir):
    all_results = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.m8'):
                m8_path = os.path.join(root, file)
                all_results.extend(process_m8_file(m8_path, root, template_dir, output_dir))
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Foldseek align out")
    parser.add_argument('--input_dir', required=True, help='.m8 file')
    parser.add_argument('--template_dir', required=True, help='PDB template')
    parser.add_argument('--output_dir', required=True, help='out')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = process_directory(args.input_dir, args.template_dir, args.output_dir)
    for result in results:
        print(result)
