#!/usr/bin/env bash

# --- CONFIGURATION ---

# Path to the Hugging Face model or a local model directory
MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

# Python executable name. Assumes it's in your PATH after activating the environment.
PYTHON_EXEC="python"

# List of GPU IDs to use for the experiments.
GPUS=(1) # Example: Use GPU 0 and 1

# Maximum number of concurrent jobs to run per GPU.
MAX_JOBS_PER_GPU=1

# --- EXPERIMENT PARAMETERS ---

# List of methods to test
METHODS=(
  "SentenceKV"
  # "FullKV"
  # "SnapKV"
  # "H2O"
  # "quest"
)

# List of KV cache capacities (max_capacity_prompts)
KV_CACHE_CAPACITIES=(1024)

# List of context lengths to evaluate
CONTEXT_LENGTHS=(
  16384
  # 32768
  # 65536
)

# List of RULER datasets to run
DATASETS=(
  "niah_single_1"
  # "niah_single_2"
  # "niah_single_3"
  # "niah_multikey_1"
  # "niah_multikey_2"
  # "niah_multivalue"
  # "niah_multiquery"
  # "vt"
  # "fwe"
  # "qa_1"
  # "qa_2"
)

# --- SCRIPT LOGIC (No need to edit below this line) ---

# Get the directory where the script is located
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
RUN_SCRIPT_PATH="${SCRIPT_DIR}/run_ruler.py"
SAVE_DIR="${SCRIPT_DIR}/results_ruler"

# Create log directory
mkdir -p "${SAVE_DIR}/logs"

# Extract the base model name for saving results
MODEL_NAME=$(basename "${MODEL_PATH}")

# --- Job Management ---

GPU_COUNT=${#GPUS[@]}
MAX_CONCURRENT_JOBS=$(( GPU_COUNT * MAX_JOBS_PER_GPU ))
COUNTER=0

# This function waits until there is a free slot to run a new job.
wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "${MAX_CONCURRENT_JOBS}" ]; do
    sleep 2
  done
}

# --- Main Loop ---

for method in "${METHODS[@]}"; do
  for capacity in "${KV_CACHE_CAPACITIES[@]}"; do
    for context_length in "${CONTEXT_LENGTHS[@]}"; do
      for data in "${DATASETS[@]}"; do

        # Define result and log file paths
        result_dir="${SAVE_DIR}/${MODEL_NAME}_${capacity}/${context_length}/${data}"
        result_file="${result_dir}/${method}.json"
        log_file="${SAVE_DIR}/logs/${method}_${capacity}_${context_length}_${data}.log"

        # If the result file already exists, skip this run
        if [ -f "${result_file}" ]; then
          echo "Skipping ${method} | capacity ${capacity} | context ${context_length} | data ${data} -> Result already exists."
          continue
        fi

        wait_for_slot

        # Assign a GPU in a round-robin fashion
        GPU_ID=${GPUS[$(( COUNTER % GPU_COUNT ))]}
        echo "Running on GPU ${GPU_ID}: method=${method}, capacity=${capacity}, context=${context_length}, data=${data}"

        # Launch the Python script in the background
        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
          "${PYTHON_EXEC}" "${RUN_SCRIPT_PATH}" \
            --method "${method}" \
            --model_path "${MODEL_PATH}" \
            --max_capacity_prompts "${capacity}" \
            --context_length "${context_length}" \
            --attn_implementation "flash_attention_2" \
            --save_dir "${SAVE_DIR}" \
            --dataset "${data}" \
            --semantic_factor 2 \
          > "${log_file}" 2>&1 &

        (( COUNTER++ ))
      done
    done
  done
done

# Wait for all background jobs to complete
echo "All experiments have been launched. Waiting for completion..."
wait
echo "All jobs completed."