export CUDA_VISIBLE_DEVICES=$1

method="sentencekv" # Support PyramidKV, SnapKV, H2O, StreamingLLM, CAM, L2Norm, ThinK
max_capacity_prompts=128 # 128,2048 in paper
attn_implementation="flash_attention_2" # Support "flash_attention_2", "sdpa", "eager".
source_path="/home/zhuy27/kv_cache/Contextual-KV-Caching/SentenceKV/LongBench/"
# model_path="meta-llama/Llama-3.1-8B-Instruct"
model_path="togethercomputer/LLaMA-2-7B-32K"
save_dir=${source_path}"results_long_bench" # path to result save_dir

/home/zhuy27/data/miniconda3x86/envs/sentenceKV/bin/python run_longbench.py \
    --method ${method} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} 
