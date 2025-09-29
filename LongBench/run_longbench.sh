#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- CONFIGURABLE PARAMETERS ---

# Specify which GPU(s) to use, e.g., "0" or "0,1"
export CUDA_VISIBLE_DEVICES=0

# List of model paths (from Hugging Face Hub or a local directory)
MODEL_PATHS=(
    "meta-llama/Llama-3.2-3B-Instruct"
    # "lmsys/longchat-7b-v1.5-32k"
)

# List of methods to test
METHODS=("SentenceKV" "FullKV" "SnapKV" "H2O" "quest")

# ! Infllm Needs to be run with different configuration


# List of datasets to evaluate on
DATASETS=("narrativeqa" "qasper" "multifieldqa_en" "hotpotqa" "2wikimqa" "musique" \
          "gov_report" "qmsum" "multi_news" "trec" "triviaqa" "samsum" \
          "passage_count" "passage_retrieval_en" "lcc" "repobench-p")

DATASETS=("narrativeqa" "qasper")

# List of KV Cache capacity values to test
CAPACITY_VALUES=(1024)

# Number of experiments to run in parallel
MAX_PARALLEL_JOBS=2

# --- SCRIPT BODY ---

# Get the directory where this script is located
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Set the save directory to a 'results' folder within the script's directory
SAVE_DIR="${SCRIPT_DIR}/results"

echo "Experiment results will be saved to: ${SAVE_DIR}"

# Loop through all configured models, methods, and capacity values
for model_path in "${MODEL_PATHS[@]}"; do
    echo "######################################################"
    echo "Running Model: ${model_path}"
    echo "######################################################"
    for method in "${METHODS[@]}"; do
        echo "===================================================="
        echo "Running Method: ${method}"
        echo "===================================================="
        for capacity in "${CAPACITY_VALUES[@]}"; do
            echo "--> Starting experiments for capacity: ${capacity}"

            # Use xargs and tr for parallel execution
            # 'tr' converts the space-separated dataset list into a newline-separated list
            # 'xargs' reads each line (dataset) and executes the command in a subshell
            echo "${DATASETS[@]}" | tr ' ' '\n' | xargs -I {} -P ${MAX_PARALLEL_JOBS} bash -c '
                # Get the current dataset name from the placeholder {}
                dataset_name="{}"
                echo "Processing dataset: ${dataset_name} for model: '"${model_path}"' ..."

                # Execute the Python script
                # Calling "python" assumes it is in the user''s PATH (from an activated virtual env)
                python "'"${SCRIPT_DIR}"'/run_longbench.py" \
                    --dataset "${dataset_name}" \
                    --method "'"${method}"'" \
                    --model_path "'"${model_path}"'" \
                    --max_capacity_prompts "'"${capacity}"'" \
                    --save_dir "'"${SAVE_DIR}"'" \
                    --attn_implementation "flash_attention_2" \
                
                echo "Finished processing dataset: ${dataset_name}"
            '

            echo "--> All experiments for capacity=${capacity} are complete."
            echo
        done
        echo "All experiments for method=${method} are complete."
        echo
    done
    echo "All experiments for model=${model_path} are complete."
    echo
done

echo "All experiments have successfully completed!"