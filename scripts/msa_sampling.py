import os
import numpy as np
import sys  # For sys.stderr
from typing import Sequence, Iterable, Optional
from tqdm import tqdm  # For progress bars


def _convert_sto_seq_to_a3m(
        query_non_gaps: Sequence[bool], sto_seq: str) -> Iterable[str]:
    """Internal helper to convert Stockholm sequence to A3M based on query gaps."""
    for is_query_res_non_gap, sequence_res in zip(query_non_gaps, sto_seq):
        if is_query_res_non_gap:
            yield sequence_res
        elif sequence_res != '-':
            yield sequence_res.lower()


def convert_stockholm_to_a3m(stockholm_format: str,
                             max_sequences: Optional[int] = None,
                             remove_first_row_gaps: bool = True) -> str:
    """Converts MSA in Stockholm format to the A3M format."""
    descriptions = {}
    sequences = {}

    # Process sequences first
    for line in stockholm_format.splitlines():
        if line.strip() and not line.startswith(('#', '//')):  # Ignore comments and terminator
            seqname, aligned_seq = line.split(maxsplit=1)
            # Stop if max_sequences reached and sequence not already added
            if max_sequences is not None and len(sequences) >= max_sequences and seqname not in sequences:
                continue
            if seqname not in sequences:
                sequences[seqname] = ''
            sequences[seqname] += aligned_seq.replace('.', '')  # Remove internal gaps (dots) for sequence processing

    # Process descriptions
    for line in stockholm_format.splitlines():
        if line.startswith('#=GS'):  # Parse description info
            columns = line.split(maxsplit=3)
            seqname, feature = columns[1:3]
            value = columns[3] if len(columns) == 4 else ''
            if feature == 'DE':  # Only interested in Description (DE) field
                if max_sequences is not None and seqname not in sequences:  # If filtered by max_sequences
                    continue
                descriptions[seqname] = value.strip()

    # Convert sto format to a3m line by line
    a3m_sequences = {}
    query_non_gaps = []
    if remove_first_row_gaps and sequences:
        # Get the first sequence to determine query gaps
        first_seq_id = next(iter(sequences))
        query_sequence_raw = sequences[first_seq_id]
        query_non_gaps = [res != '-' for res in query_sequence_raw]

    for seqname, sto_sequence_raw in sequences.items():
        # Apply query gap removal if requested
        if remove_first_row_gaps:
            out_sequence = ''.join(
                _convert_sto_seq_to_a3m(query_non_gaps, sto_sequence_raw))
        else:
            out_sequence = sto_sequence_raw
        a3m_sequences[seqname] = out_sequence

    fasta_chunks = (f">{k} {descriptions.get(k, '')}\n{a3m_sequences[k]}"
                    for k in a3m_sequences)
    return '\n'.join(fasta_chunks) + '\n'


def process_uniref90_sto_to_a3m(sto_file_path: str, output_a3m_path: str) -> bool:
    """Reads a Stockholm file, converts it to A3M, and saves it."""
    try:
        with open(sto_file_path, 'r') as f:
            sto_str = f.read()

        a3m_str = convert_stockholm_to_a3m(sto_str)

        # Ensure output directory exists before writing
        os.makedirs(os.path.dirname(output_a3m_path), exist_ok=True)
        with open(output_a3m_path, 'w') as f:
            f.write(a3m_str)
        return True
    except Exception as e:
        print(f"Error converting {sto_file_path} to A3M: {e}", file=sys.stderr)
        return False


# --- Core Masking Logic Functions ---

def process_npz(npz_path, threshold=0.5):
    """Processes the NPZ file to get the binary matrix and column sum probabilities."""
    data = np.load(npz_path)
    if 'deviation' not in data:
        raise KeyError(f"'deviation' key not found in NPZ file: {npz_path}. Available keys: {list(data.keys())}")

    matrix = data['deviation']
    # Ensure matrix is 2D, if it's 1D (single residue prediction) convert it
    if matrix.ndim == 1:
        matrix = np.expand_dims(matrix, axis=1)  # Makes it (L, 1)

    matrix_binary = (matrix > threshold).astype(int)
    prob_sum = np.sum(matrix_binary, axis=1)  # Sum over columns for each row (should be (L,) now)
    return matrix_binary, prob_sum


def sliding_window(prob_sum, window_size=3, top_percent=0.20):
    """Applies a sliding window to prob_sum and returns top windows."""
    L = len(prob_sum)
    if L < window_size:
        return []
    window_sums = [(i, np.sum(prob_sum[i:i + window_size])) for i in range(L - window_size + 1)]
    window_sums.sort(key=lambda x: x[1], reverse=True)

    num_top_windows = max(1, int(len(window_sums) * top_percent))
    top_windows = window_sums[:num_top_windows]
    return top_windows


def merge_windows(top_windows, window_size=3):
    """Merges overlapping or adjacent windows."""
    if not top_windows:
        return []
    merged_windows = []
    top_windows.sort(key=lambda x: x[0])

    current_start = top_windows[0][0]
    current_end = top_windows[0][0] + window_size - 1

    for i in range(1, len(top_windows)):
        start, _ = top_windows[i]
        end = start + window_size - 1
        if start <= current_end + 1:
            current_end = max(current_end, end)
        else:
            merged_windows.append((current_start, current_end))
            current_start = start
            current_end = end

    merged_windows.append((current_start, current_end))
    return merged_windows


def firetrain_mask_for_long_fragment(start, end):
    """
    Generates 'firetrain' like mask positions for long fragments (length > 6).
    The masks stop generating when the left and right 3-residue windows touch or overlap.
    """
    mask_positions = []

    left_ptr = start
    right_ptr = end

    while (left_ptr + 2) < (right_ptr - 2):
        left_mask = list(range(left_ptr, left_ptr + 3))
        right_mask = list(range(right_ptr - 2, right_ptr + 1))
        mask_positions.append(left_mask + right_mask)

        left_ptr += 1
        right_ptr -= 1

    if right_ptr - left_ptr + 1 >= 3:
        mask_positions.append(list(range(left_ptr, right_ptr + 1)))

    return mask_positions


def read_a3m_sequences(a3m_path):
    sequences = {}
    with open(a3m_path, 'r') as f:
        seq_id = None
        sequence = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id:
                    processed_seq = ''.join([residue for residue in sequence if residue.isupper() or residue == "-"])
                    if processed_seq:
                        sequences[seq_id] = processed_seq
                seq_id = line[1:]
                sequence = ""
            else:
                sequence += line
        if seq_id:
            processed_seq = ''.join([residue for residue in sequence if residue.isupper() or residue == "-"])
            if processed_seq:
                sequences[seq_id] = processed_seq
    return sequences


# --- MSA Splitting and Renaming Functions ---

def filter_sequences_by_length(sequences):
    sequences_list = list(sequences.items())
    if not sequences_list:
        print("Warning: No sequences found to filter.", file=sys.stderr)
        return {}

    main_sequence_length = len(sequences_list[0][1])
    filtered_sequences = {
        seq_id: seq for seq_id, seq in sequences.items()
        if len(seq) == main_sequence_length
    }
    return filtered_sequences


def split_msa_by_depth(sequences, depth_split_sizes):

    sequences_list = list(sequences.items())
    msa_parts = {}
    total_sequences = len(sequences_list)

    generated_depths = set()

    sorted_depths = sorted(list(set(d for d in depth_split_sizes if d > 0)))

    for depth_label, target_depth in enumerate(sorted_depths, start=1):
        actual_depth_to_cut = min(target_depth, total_sequences)

        if actual_depth_to_cut in generated_depths:
            continue

        msa_part = dict(sequences_list[:actual_depth_to_cut])

        generated_depths.add(actual_depth_to_cut)

        msa_parts[f"subset_{actual_depth_to_cut}"] = msa_part
    return msa_parts


def save_msa_subset(msa_part, output_dir, sample_folder_name,
                    current_target_file_index):
    """
    Saves a single MSA subset to a new A3M file with the simplified naming:
    sample_folder_name_current_target_file_index.a3m
    """
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir,
                                   f"{sample_folder_name}_{current_target_file_index:04d}.a3m")  # Use current_target_file_index
    try:
        with open(output_filename, 'w') as f:
            for seq_id, seq in msa_part.items():
                f.write(f">{seq_id}\n{seq}\n")
    except IOError as e:
        print(f"Error saving MSA subset to {output_filename}: {e}", file=sys.stderr)


# --- Combined Workflow Logic ---

def process_and_mask_a3m(a3m_path, fragments, sample_folder_name, protein_folder_path, current_target_file_index,
                         depth_split_sizes):
    """
    Applies masking to a single A3M file, then performs depth sampling and saves.
    Returns the updated index for the current target.
    """
    sequences_dict = read_a3m_sequences(a3m_path)

    # Output target directory is now sub_msas
    output_target_dir = os.path.join(protein_folder_path, "sub_msas")
    os.makedirs(output_target_dir, exist_ok=True)

    # Process each merged fragment
    for frag_idx, (start, end) in enumerate(fragments):
        fragment_length = end - start + 1
        mask_combinations_for_this_fragment = []

        if fragment_length > 6:
            mask_combinations_for_this_fragment = firetrain_mask_for_long_fragment(start, end)
        else:
            mask_combinations_for_this_fragment = [[i for i in range(start, end + 1)]]

        if not mask_combinations_for_this_fragment:
            print(
                f"Warning: No mask combinations generated for fragment {start}-{end} in {os.path.basename(a3m_path)}. Skipping.",
                file=sys.stderr)  # Added filename to warning
            continue

        for mask_idx, mask_positions in enumerate(mask_combinations_for_this_fragment):
            current_sequences_for_masking = {}
            seq_counter_in_file = 0
            for seq_id, seq_content in sequences_dict.items():
                if seq_counter_in_file == 0:
                    current_sequences_for_masking[seq_id] = seq_content
                else:
                    modified_seq_list = list(seq_content)
                    for pos in mask_positions:
                        # Ensure position is within sequence bounds and not an existing gap
                        if 0 <= pos < len(modified_seq_list) and modified_seq_list[pos] != "-":
                            modified_seq_list[pos] = "X"
                    current_sequences_for_masking[seq_id] = "".join(modified_seq_list)
                seq_counter_in_file += 1

            # Perform depth sampling on this newly masked MSA
            filtered_sequences = filter_sequences_by_length(current_sequences_for_masking)
            if not filtered_sequences:
                print(
                    f"Warning: No sequences remained after filtering for length in fragment {start}-{end}, mask combo {mask_idx + 1} for {os.path.basename(a3m_path)}. Skipping depth sampling.",
                    file=sys.stderr)  # Added filename to warning
                continue

            msa_parts = split_msa_by_depth(filtered_sequences, depth_split_sizes)

            if not msa_parts:
                print(
                    f"Warning: No MSA parts generated after depth splitting for fragment {start}-{end}, mask combo {mask_idx + 1} for {os.path.basename(a3m_path)}. Skipping saving.",
                    file=sys.stderr)
                continue

            for label, msa_part in msa_parts.items():
                save_msa_subset(msa_part, output_target_dir, sample_folder_name,
                                current_target_file_index)  # Use current_target_file_index
                current_target_file_index += 1

    return current_target_file_index


def main_workflow(base_input_dir, npz_filename="flex.npz", threshold=0.3, window_size=3, top_percent=0.2,
                  depth_split_sizes=None):
    """
    Main workflow to process each protein folder: find flex.npz, convert .sto to .a3m,
    apply masking to MSAs, then perform depth sampling and save, ensuring continuous indexing PER TARGET.
    """
    if depth_split_sizes is None:
        depth_split_sizes = [16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120]

    relevant_subfolders = []
    for entry in os.listdir(base_input_dir):
        subfolder_path = os.path.join(base_input_dir, entry)
        if os.path.isdir(subfolder_path):
            msa_subfolder_path = os.path.join(subfolder_path, "msas")
            if os.path.exists(msa_subfolder_path) and (
                    os.path.exists(os.path.join(msa_subfolder_path, "bfd_hits.a3m")) or
                    os.path.exists(os.path.join(msa_subfolder_path, "uniref30_hits.a3m")) or
                    os.path.exists(os.path.join(msa_subfolder_path, "uniref90_hits.sto"))
            ):
                relevant_subfolders.append(entry)

    if not relevant_subfolders:
        print(
            f"Error: No relevant protein subfolders with 'msas' subdirectory and expected MSA files found in {base_input_dir}. Exiting.",
            file=sys.stderr)
        return

    relevant_subfolders.sort()

    for sample_folder_name in tqdm(relevant_subfolders, desc="Processing Proteins"):  # Use tqdm for overall progress
        protein_folder_path = os.path.join(base_input_dir, sample_folder_name)
        msa_input_path_for_protein = os.path.join(protein_folder_path, "msas")

        # Reset index for each new protein target
        current_target_file_index = 1

        current_npz_path = os.path.join(protein_folder_path, npz_filename)
        if not os.path.exists(current_npz_path):
            print(
                f"Error: NPZ file not found: {current_npz_path}. Skipping processing for folder '{sample_folder_name}'.",
                file=sys.stderr)
            continue

        try:
            matrix_binary, prob_sum = process_npz(current_npz_path, threshold)
            top_windows = sliding_window(prob_sum, window_size, top_percent)
            merged_fragments = merge_windows(top_windows, window_size)

            if not merged_fragments:
                print(
                    f"Warning: No flexible fragments found above threshold for {current_npz_path}. Skipping masking for folder '{sample_folder_name}'.",
                    file=sys.stderr)
                continue

        except KeyError as e:
            print(f"Error processing NPZ file {current_npz_path}: {e}. Skipping folder '{sample_folder_name}'.",
                  file=sys.stderr)
            continue
        except Exception as e:
            print(
                f"General error with NPZ processing for {current_npz_path}: {e}. Skipping folder '{sample_folder_name}'.",
                file=sys.stderr)
            continue

        uniref90_sto_path = os.path.join(msa_input_path_for_protein, "uniref90_hits.sto")
        uniref90_a3m_path = os.path.join(msa_input_path_for_protein, "uniref90_hits.a3m")

        msa_files_to_process = []

        if os.path.exists(uniref90_sto_path):
            if not os.path.exists(uniref90_a3m_path):
                if process_uniref90_sto_to_a3m(uniref90_sto_path, uniref90_a3m_path):
                    msa_files_to_process.append("uniref90_hits.a3m")
                else:
                    print(f"Error: Failed to convert {uniref90_sto_path}. It will not be included.", file=sys.stderr)
            else:
                msa_files_to_process.append("uniref90_hits.a3m")

        # Add BFD.a3m and UniRef30.a3m if they exist
        for f in ["bfd_hits.a3m", "uniref30_hits.a3m"]:
            full_path_to_check = os.path.join(msa_input_path_for_protein, f)
            if os.path.exists(full_path_to_check):
                msa_files_to_process.append(f)
            else:
                pass  # Suppress non-error print for missing files

        if not msa_files_to_process:
            print(
                f"Warning: No valid MSA files (bfd_hits.a3m, uniref30_hits.a3m, or converted uniref90_hits.a3m) found in {msa_input_path_for_protein} for {sample_folder_name}. Skipping masking for this protein.",
                file=sys.stderr)
            continue

        for specific_msa_file_name in msa_files_to_process:
            msa_path_to_mask = os.path.join(msa_input_path_for_protein, specific_msa_file_name)

            current_target_file_index = process_and_mask_a3m(
                msa_path_to_mask,
                merged_fragments,
                sample_folder_name,
                protein_folder_path,
                current_target_file_index,
                depth_split_sizes
            )


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MSA folder and apply masking strategy.")
    parser.add_argument(
        "--input_dir", required=True, type=str,
        help="The base input directory containing protein folders with `msas` and `flex.npz`."
    )
    parser.add_argument(
        "--npz_name", default="flex.npz", type=str,
        help="The filename of the .npz file inside each protein folder (default: flex.npz)"
    )
    parser.add_argument(
        "--threshold", default=0.3, type=float,
        help="Threshold for deviation binarization (default: 0.5)"
    )
    parser.add_argument(
        "--window_size", default=3, type=int,
        help="Sliding window size for fragment detection (default: 3)"
    )
    parser.add_argument(
        "--top_percent", default=0.2, type=float,
        help="Top fraction of fragments to select (default: 0.2)"
    )
    parser.add_argument(
        "--depths", type=str,
        help="Comma-separated list of depths for MSA subsampling, e.g., '16,32,64,128'"
    )

    args = parser.parse_args()

    if args.depths:
        depth_split_sizes = [int(x) for x in args.depths.split(',')]
    else:
        depth_split_sizes = None

    main_workflow(
        base_input_dir=args.input_dir,
        npz_filename=args.npz_name,
        threshold=args.threshold,
        window_size=args.window_size,
        top_percent=args.top_percent,
        depth_split_sizes=depth_split_sizes
    )
