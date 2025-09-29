import os
import sys
# Add parent directory to system path to import modules like 'models'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import random
import argparse
from tqdm import tqdm
import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Import monkey patches and enabling functions for various methods
from models.pyramidkv.monkeypatch import replace_llama, replace_mistral
from models.quest.quest_attention import enable_quest_attention_eval
from models.quest.llama import enable_tuple_kv_cache_for_llama
from models.quest.mistral import enable_tuple_kv_cache_for_mistral

# --- Inf-LLM Imports ---
from omegaconf import OmegaConf
from models.inf_llm.utils import patch_hf, GreedySearch 

# RULER dataset configurations
datasets = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
            "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt", "qa_1", "qa_2"]

dataset2maxlen = {
    "niah_single_1": 64, "niah_single_2": 64, "niah_single_3": 64,
    "niah_multikey_1": 64, "niah_multikey_2": 64, "niah_multikey_3": 64,
    "niah_multiquery": 64, "niah_multivalue": 64, "cwe": 64, "fwe": 64,
    "vt": 64, "qa_1": 64, "qa_2": 64,
}

# Model max length configurations, referenced and expanded from run_longbench.py
model2maxlen = {
    "Llama-2-7b-chat-hf": 3950,
    "Llama-3-8B-Instruct": 7950, 
    "Meta-Llama-3-70B-Instruct": 7950,
    "Meta-Llama-3-8B-Instruct-32k": 31500,
    "Llama-2-7B-32K-Instruct": 31500,
    "Mistral-7B-Instruct-v0.2": 31500,
    "Mistral-7B-Instruct-v0.1": 31500,
    "c-8B-Instruct": 31500,
    "Llama-3.1-8B-Instruct": 128000,
    "longchat-7b-v1.5-32k": 31500,
    "Meta-Llama-3-8B-1M": 140000,
    "Meta-Llama-3-8B": 7950,
}

# --- Inf-LLM Helper Functions ---
def get_model_and_tokenizer_infllm(config, args):
    """InfLLM's model loading function."""
    tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path,
            use_fast=args.use_fast_tokenizer,
            padding_side="left"
        )
    model = AutoModelForCausalLM.from_pretrained(config.path, torch_dtype=torch.float16, device_map="auto", 
                                                cache_dir=os.path.join(os.path.expanduser("~"), "huggingface"),
                                                attn_implementation=args.attn_implementation)
    model = patch_hf(model, config.type, **config)
    return model, tokenizer

def post_process_infllm(pred, model_name, dataset):
    """Post-processing function for InfLLM outputs."""
    if "qwen" in model_name: # model_name is conv_type in this script
        pred = pred.split("<|im_end|>")[0]
    return pred
# --- Inf-LLM Helper Functions End ---


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def build_chat(prompt):
    """Build chat template for Llama2."""
    return f"[INST] {prompt} [/INST]"

def build_chat_llama3(prompt):
    """Build chat template for Llama3."""
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

def main(args, model, tokenizer):
    """Main evaluation function."""
    print("Loading data...")
    
    test_data = []
    with open(args.data_file) as fp:
        for line in fp:
            example = json.loads(line)
            prompt = example["input"]
            
            # Automatically apply chat template based on model path
            if args.use_chat_format:
                if "llama-3" in args.model_path.lower() or "llama3" in args.model_path.lower():
                    prompt = build_chat_llama3(prompt)
                elif "llama-2" in args.model_path.lower() or "llama2" in args.model_path.lower():
                    prompt = build_chat(prompt)
            
            example["prompt"] = prompt
            test_data.append(example)

    model_path = args.model_path
    model_max_len = -1
    for key in model2maxlen:
        if key in model_path:
            model_max_len = model2maxlen[key]
            break
    if model_max_len == -1:
        print("Warning: Model max length not found in config, using default 4096.")
        model_max_len = 4096

    print(f"Model: {model_path}, Model Max Length: {model_max_len}")
        
    if args.max_num_examples and len(test_data) > args.max_num_examples:
        test_data = random.sample(test_data, args.max_num_examples) if args.sample_method == "random" else test_data[:args.max_num_examples]
    
    # Extract data fields
    prompts = [ex["prompt"] for ex in test_data]
    answers_list = [ex["outputs"] for ex in test_data]
    lengths = [ex["length"] for ex in test_data]

    print("Finished loading data. Starting generation...")
    
    model_name = model_path.split("/")[-1]
    save_path = os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", str(args.context_length), args.dataset)
    os.makedirs(save_path, exist_ok=True)
     
    with open(os.path.join(save_path, f"{args.method}.json"), "w") as fout:
        for i in tqdm(range(0, len(prompts), args.eval_batch_size)):
            # Prepare batch data
            batch_prompts = prompts[i:i+args.eval_batch_size]
            batch_answers = answers_list[i:i+args.eval_batch_size]
            batch_lengths = lengths[i:i+args.eval_batch_size]
            
            tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
            batch_input_ids = tokenized_prompts.input_ids

            # Handle inputs longer than model's max length
            if batch_input_ids.shape[1] > model_max_len:
                print(f"Warning: Input length {batch_input_ids.shape[1]} exceeds model max length {model_max_len}. Truncating.")
                half = int(model_max_len / 2)
                truncated_ids = torch.cat(
                    [batch_input_ids[0, :half], batch_input_ids[0, -half:]]
                )
                batch_input_ids = truncated_ids.unsqueeze(0)
            
            # Calculate token budget
            if args.max_capacity_prompts != -1:
                max_capacity_prompts = args.max_capacity_prompts
            elif args.max_capacity_prompts_ratio != -1:
                max_capacity_prompts = round(batch_input_ids.shape[1] * args.max_capacity_prompts_ratio)
            else:
                max_capacity_prompts = 2048 # default value

            # Configure model parameters for different methods
            if args.method.lower() in ["snapkv", "h2o"]:
                layers = len(model.model.layers)
                for i in range(layers):
                    model.model.layers[i].self_attn.config.window_size = 32
                    model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts
                    model.model.layers[i].self_attn.config.kernel_size = 7
                    model.model.layers[i].self_attn.config.pooling = "maxpool"
            
            elif args.method.lower() == "sentencekv":
                for layer in model.model.layers:
                    if hasattr(layer.self_attn, 'sentence_cache'):
                        layer.self_attn.sentence_cache = []
                    if hasattr(layer.self_attn, 'head_mean_k_tensor'):
                        layer.self_attn.head_mean_k_tensor = None
                    if hasattr(layer.self_attn, 'head_token_indices'):
                        layer.self_attn.head_token_indices = None
                    if hasattr(layer.self_attn, 'kv_seq_len'):
                        layer.self_attn.kv_seq_len = 0
                    if hasattr(layer.self_attn, 'token_budget'):
                        layer.self_attn.token_budget = max_capacity_prompts
                    if hasattr(layer.self_attn, 'semantic_factor'):
                        layer.self_attn.semantic_factor = args.semantic_factor
                    if hasattr(layer.self_attn, 'semantic_budget'):
                        layer.self_attn.semantic_budget = int(args.semantic_factor * max_capacity_prompts)
                
                if hasattr(model.model, '_init_cache') and callable(model.model._init_cache):
                    model.model._init_cache()

            context_length = batch_input_ids.shape[-1]
            output_max_len = dataset2maxlen.get(args.dataset, 64)

            # --- Generation Logic ---
            if args.method.lower() == "infllm":
                searcher = GreedySearch(model, tokenizer)
                output = searcher.generate(
                    input_ids=batch_input_ids[0],
                    max_length=output_max_len,
                    chunk_size=args.chunk_size,
                )
                pred = post_process_infllm(output[0], args.conv_type, args.dataset)
                batch_generations = [pred]
                searcher.clear()

            elif args.method.lower() == "quest":
                with torch.no_grad():
                    output = model(input_ids=batch_input_ids, past_key_values=None, use_cache=True)
                    past_key_values = output.past_key_values
                    
                    pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    generated_content = [pred_token_idx.item()]
                    
                    for _ in range(output_max_len - 1):
                        outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values, use_cache=True)
                        past_key_values = outputs.past_key_values
                        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                        generated_content.append(pred_token_idx.item())
                        if pred_token_idx.item() == tokenizer.eos_token_id:
                            break
                    
                    full_output_ids = batch_input_ids[0].tolist() + generated_content
                    output_tensor = torch.tensor([full_output_ids], device=batch_input_ids.device)
                    batch_generations = tokenizer.batch_decode([output_tensor[0][context_length:]], skip_special_tokens=True)

            else: # Default generation logic
                output = model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=torch.ones_like(batch_input_ids),
                    max_new_tokens=output_max_len,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    eos_token_id=tokenizer.eos_token_id
                )
                batch_generations = tokenizer.batch_decode([output[0][context_length:]], skip_special_tokens=True)

            torch.cuda.empty_cache()

            # Save results
            for j in range(len(batch_generations)):
                example = {
                    "answers": batch_answers[j],
                    "pred": batch_generations[j],
                    "length": batch_lengths[j],
                }
                fout.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # General arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="results_ruler")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True)
    parser.add_argument("--max_num_examples", type=int, default=None)
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"])
    parser.add_argument("--eval_batch_size", type=int, default=1)
    
    # RULER specific arguments
    parser.add_argument("--model_template", type=str, default="llama-3", help="The model template to use for finding data file.")
    parser.add_argument("--context_length", type=int, default=16384, help="Context length for evaluation.")
    parser.add_argument("--dataset", type=str, default="niah_single_1", choices=datasets)
    parser.add_argument("--use_chat_format", action="store_true", help="Use chat format for prompts.")

    # Method-specific arguments
    parser.add_argument("--method", type=str, default="SentenceKV", 
                        choices=["SentenceKV", "FullKV", "SnapKV", "H2O", "quest", "infllm"])
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", 
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--max_capacity_prompts", type=int, default=2048)
    parser.add_argument("--max_capacity_prompts_ratio", type=float, default=-1)
    parser.add_argument("--semantic_factor", type=float, default=3, help="For SentenceKV")
    parser.add_argument("--token_budget", type=int, default=None, help="For Quest")

    # Inf-LLM specific arguments
    parser.add_argument("--chunk_size", type=int, default=32, help="For InfLLM")
    parser.add_argument("--config_path", type=str, default="models/inf_llm/config", help="Path to InfLLM YAML config file directory")
    parser.add_argument("--conv_type", type=str, default="llama-3.1-inf-llm.yaml", help="YAML config file for InfLLM")


    args = parser.parse_args()
    args.save_dir = os.path.join(os.path.dirname(__file__), args.save_dir)
    args.token_budget = args.max_capacity_prompts 

    set_seed(args.seed)
    
    model, tokenizer = None, None
    huggingface_cache_dir = os.path.join(os.path.expanduser("~"), "huggingface")

    # --- Model & Tokenizer Loading ---
    print(f"Loading model and tokenizer for method: {args.method}")
    
    if args.method.lower() == "infllm":
        config_file_path = os.path.join(os.path.dirname(__file__), "..", args.config_path, args.conv_type)
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Inf-LLM config not found at: {config_file_path}")
        conf = OmegaConf.load(config_file_path)
        if not hasattr(conf.model, "tokenizer_path"):
            conf.model.tokenizer_path = conf.model.path
        model, tokenizer = get_model_and_tokenizer_infllm(conf.model, args)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            use_fast=args.use_fast_tokenizer,
            padding_side="left",
            cache_dir=huggingface_cache_dir
        )
        
        model_dtype = torch.float16 if args.method.lower() != 'quest' else torch.bfloat16
        
        if args.method.lower() in ["snapkv", "h2o"]:
            replace_llama(args.method.lower())
            replace_mistral(args.method.lower())    
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, torch_dtype=model_dtype, device_map="auto",
                attn_implementation=args.attn_implementation, cache_dir=huggingface_cache_dir
            )
        elif args.method.lower() == "fullkv":
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, torch_dtype=model_dtype, device_map="auto",
                attn_implementation=args.attn_implementation, cache_dir=huggingface_cache_dir
            )
        elif args.method.lower() == "sentencekv":
            config = AutoConfig.from_pretrained(args.model_path, cache_dir=huggingface_cache_dir)
            from models.sentencekv.custom_llama_sentencekv import LlamaForCausalLM 
            punctuations = ['.','!','?','. ','! ','? ','...','\n\n','\n\n\n', '。','！','？','。。']
            punctuations_ids = [tokenizer.convert_tokens_to_ids(p) for p in punctuations]
            config.punctuations_ids = punctuations_ids
            model = LlamaForCausalLM.from_pretrained(
                args.model_path, torch_dtype=model_dtype, device_map="auto", config=config,
                attn_implementation=args.attn_implementation, cache_dir=huggingface_cache_dir
            )
        elif args.method.lower() == "quest":
            if "llama" in args.model_path.lower() or "longchat" in args.model_path.lower():
                enable_tuple_kv_cache_for_llama()
            if "mistral" in args.model_path.lower():
                enable_tuple_kv_cache_for_mistral()
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, trust_remote_code=True, torch_dtype=model_dtype, 
                device_map="auto", cache_dir=huggingface_cache_dir
            )
            enable_quest_attention_eval(model, args)
    
    # Configure tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.eval()
    
    # Set data file path
    base_data_dir = os.path.join(os.path.dirname(__file__), "data")
    args.data_file = os.path.join(base_data_dir, args.model_template, str(args.context_length), args.dataset, "validation.jsonl")
    
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found at: {args.data_file}")
    
    print(f"Working on context length {args.context_length}, max_capacity_prompts: {args.max_capacity_prompts}, dataset: {args.dataset}")
    
    # Call the main function
    main(args, model, tokenizer)