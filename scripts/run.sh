#!/bin/bash

source "$(dirname "$0")/config.sh"
export CUDA_VISIBLE_DEVICES=1
#----------------------*** Search MSA ***----------------------
module load anaconda
source activate AFsample

python scripts/search_MSA.py \
  --pdb_dir "$pdb_dir" \
  --fasta_dir "$fasta_dir" \
  --msa_out_dir "$msa_out_dir" \
  --flagfile "$flagfile" \
  --num_threads "$num_threads"

#-------------------*** Structure Profile ***-------------------
module load anaconda
source activate foldseek

bash scripts/structure_profile1.sh \
  --input_base_dir "$pdb_dir" \
  --target_db "$target_db" \
  --output_dir_base "$msa_out_dir/structure_profile_temp1" \
  --tmp_dir_base "$msa_out_dir/structure_profile_temp1" \
  --filter_list "$filter_list" \
  --req_count 200 \
  --max_jobs 10

module load anaconda
source activate /home/data/user/dongl/sing/pytorch


python scripts/sp.py \
  --input_m8_dir "$msa_out_dir/structure_profile_temp1" \
  --template_pdb_dir "$template_dir" \
  --query_pdb_dir "$pdb_dir" \
  --tmalign_path "$BASE_DIR/scripts" \
  --final_output_dir "$msa_out_dir" \
  --filter_list "$filter_list" \
  --max_workers 1

#python scripts/copy_pdb.py \
#  --input_dir "$msa_out_dir/structure_profile_temp1" \
#  --template_dir "$template_dir" \
#  --output_dir "$msa_out_dir/structure_profile_temp2"
#
#python scripts/structure_profile2.py \
#  --tmalign_path "$BASE_DIR/scripts" \
#  --query_dir "$pdb_dir" \
#  --target_dir "$msa_out_dir/structure_profile_temp2" \
#  --align_dir "$msa_out_dir/structure_profile_temp3" \
#  --max_workers 1 \
#  --filter_list "$filter_list"
#
#python scripts/structure_profile3.py \
#  --input_dir "$msa_out_dir/structure_profile_temp3" \
#  --output_dir "$msa_out_dir" \
#  --remp_dir1 "$msa_out_dir/structure_profile_temp1" \
#  --remp_dir2 "$msa_out_dir/structure_profile_temp2" \
#  --remp_dir3 "$msa_out_dir/structure_profile_temp3"

#-------------------*** MSA embedding ***-------------------
python "$BASE_DIR"/flexible_residue/MSA_embedding/run_MSA_embeddings.py \
  --input_base_dir "$msa_out_dir" \
  --output_base_dir "$msa_out_dir"

#-------------------*** Predict FlexRes ***-------------------
python scripts/predict_FlexRes.py \
  --input "$filter_list" \
  --output "$msa_out_dir" \
  --model "$BASE_DIR/flexible_residue" \
  --msa_folder "$msa_out_dir" \
  --pdb_folder "$pdb_dir" \
  --template_folder "$msa_out_dir" \
  --process 1 \

#-------------------*** MSA subsample ***-------------------

python scripts/msa_sampling.py \
  --input_dir "$msa_out_dir" \
  --npz_name flex.npz \
  --threshold 0.3 \
  --window_size 3 \
  --top_percent 0.20 \
  --depths "16,32,64,128,256,512,1024,1536,2048,2560,3072,3584,4096,4608,5120"

#-------------------*** predicted conformation ***-------------------

module load anaconda
source activate AFsample

python scripts/predict_multiple_conformations.py \
  --input_base_dir "$msa_out_dir" \
  --output_base_dir "$msa_out_dir" \
  --folders_file "$filter_list" \
  --num_threads 1 \
  --af2_dir "$BASE_DIR/af_multiple_conformation"



