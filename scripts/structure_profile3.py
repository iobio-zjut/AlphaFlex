import os
import shutil
import numpy as np
from tqdm import tqdm
import argparse

def process_npz_files(input_dir, output_dir, remp_dir1, remp_dir2, remp_dir3):
    for root, dirs, files in os.walk(input_dir):
        subfolder_name = os.path.basename(root)

        if root == input_dir:
            continue

        all_arrays = []
        for file in tqdm(files, desc=f"Processing files in {subfolder_name}"):
            if file.endswith(".npz"):
                file_path = os.path.join(root, file)
                data = np.load(file_path)
                for array_name in data.files:
                    array = data[array_name]
                    if array.ndim == 2 and array.shape[1] == 1:
                        all_arrays.append(array)
                data.close()

        if not all_arrays:
            continue

        stacked_arrays = np.hstack(all_arrays)
        result_array = np.zeros((stacked_arrays.shape[0], 1), dtype=float)

        for i in range(stacked_arrays.shape[0]):
            row = stacked_arrays[i, :]
            valid_values = row[row > 2]
            if valid_values.size > 0:
                result_array[i, 0] = valid_values.mean()
            else:
                result_array[i, 0] = -1

        output_subfolder = os.path.join(output_dir, subfolder_name)
        os.makedirs(output_subfolder, exist_ok=True)

        output_path = os.path.join(output_subfolder, "structure_profile.npz")
        np.savez(output_path, result=result_array)

    for remp in [remp_dir1, remp_dir2, remp_dir3]:
        if os.path.exists(remp):
            shutil.rmtree(remp)

def main():
    parser = argparse.ArgumentParser(description="Process .npz files and compute structure profile.")
    parser.add_argument('--input_dir', required=True, help="temp2")
    parser.add_argument('--output_dir', required=True, help="out")
    parser.add_argument('--remp_dir1', required=True, help="temp1")
    parser.add_argument('--remp_dir2', required=True, help="temp2")
    parser.add_argument('--remp_dir3', required=True, help="temp3")

    args = parser.parse_args()

    process_npz_files(args.input_dir, args.output_dir, args.remp_dir1, args.remp_dir2, args.remp_dir3)

if __name__ == '__main__':
    main()
