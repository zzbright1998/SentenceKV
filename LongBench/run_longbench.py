import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import random
import argparse
from models.pyramidkv.monkeypatch import replace_llama, replace_mistral
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from models.quest.quest_attention import enable_quest_attention_eval
from models.quest.llama import enable_tuple_kv_cache_for_llama
from models.quest.mistral import enable_tuple_kv_cache_for_mistral

from omegaconf import OmegaConf
from models.inf_llm.utils import patch_hf, GreedySearch 



datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", 
            "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", 
            "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64
}

model2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}

model2maxlen = {
    "Llama-2-7b-chat-hf": 3950,
    "Llama-3-8B-Instruct": 7950, 
    "Meta-Llama-3-70B-Instruct": 7950,
    "Meta-Llama-3-8B-Instruct-32k": 31500,
    "Llama-2-7B-32K-Instruct": 31500,
    "Mistral-7B-Instruct-v0.2": 31500,
    "Mistral-7B-Instruct-v0.1": 31500,
    "c-8B-Instruct": 31500,
    "Llama-3.1-8B-Instruct": 31500,
    "longchat-7b-v1.5-32k": 31500,
    "Llama-3.2-3B-Instruct": 31500,
    "Meta-Llama-3-8B": 7950,
}

# --- InfLLM Helper Functions Start ---
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
    if model_name == "qwen":
        pred = pred.split("<|im_end|>")[0]

    if dataset == "samsum":
        pred = pred.split("\n")[0].strip()

    return pred
# --- InfLLM Helper Functions End ---

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def build_chat(prompt):
    prompt = f"[INST] {prompt} [/INST]"
    return prompt

def build_chat_llama3(prompt):
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return prompt

def main(args, model, tokenizer):
    print("Loading data...")
    
    test_data, prompts, inputs, contexts, answerss, lengths, datasets, languages, all_classess, _ids = [], [], [], [], [], [], [], [], [], []
    input_max_len = 0
    model_path = args.model_path
    
    for key in model2maxlen:
        if key in model_path:
            model_max_len = model2maxlen[key]
    
    output_max_len = dataset2maxlen[args.dataset]
    
    with open(args.data_file) as fp:
        for line in fp:
            example = json.loads(line)
            length = example["length"]
            if length > input_max_len: input_max_len = length
            
            template = model2prompt[args.dataset]
            prompt = template.format(**example)
            
            if "llama2" in args.model_path.lower():
                prompt = build_chat(prompt)
            elif "llama-3" in args.model_path.lower(): 
                prompt = build_chat_llama3(prompt)
                
            example["prompt"] = prompt
            test_data.append(example)
        
    print(f"Max Length is {input_max_len}")
        
    if args.max_num_examples and len(test_data) > args.max_num_examples:
        test_data = random.sample(test_data, args.max_num_examples) if args.sample_method == "random" else test_data[:args.max_num_examples]
    
    for example in test_data:
        prompts.append(example["prompt"])
        inputs.append(example.get("input", ""))
        contexts.append(example.get("context", ""))
        answerss.append(example["answers"])
        lengths.append(example["length"])
        datasets.append(example.get("dataset", args.dataset))
        languages.append(example.get("language", "en"))
        all_classess.append(example["all_classes"])
        _ids.append(example.get("_id", ""))

    print("Finished loading data. Starting generation...")
    
    model_name = model_path.split("/")[-1]
    save_path = os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset)
    os.makedirs(save_path, exist_ok=True)
     
    with open(os.path.join(save_path, f"{args.method}.json"), "w") as fout:
        for i in tqdm(range(0, len(prompts), args.eval_batch_size)):
            batch_prompts = prompts[i:i+args.eval_batch_size]
            batch_answerss = answerss[i:i+args.eval_batch_size]
            batch_lengths = lengths[i:i+args.eval_batch_size]
            batch_datasets = datasets[i:i+args.eval_batch_size]
            batch_languages = languages[i:i+args.eval_batch_size]
            batch_all_classess = all_classess[i:i+args.eval_batch_size]
            batch__ids = _ids[i:i+args.eval_batch_size]
            
            tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
            batch_input_ids = tokenized_prompts.input_ids

            if len(batch_input_ids[0]) > model_max_len:
                half = int(model_max_len/2)
                prompt = tokenizer.decode(batch_input_ids[0][:half], skip_special_tokens=True) + tokenizer.decode(batch_input_ids[0][-half:], skip_special_tokens=True)
                tokenized_prompts = tokenizer(prompt, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
                batch_input_ids = tokenized_prompts.input_ids

            if args.max_capacity_prompts != -1:
                max_capacity_prompts = args.max_capacity_prompts
            elif args.max_capacity_prompts_ratio != -1:
                max_capacity_prompts = round(batch_input_ids.shape[1] * args.max_capacity_prompts_ratio)
            
            # Configure parameters for different methods
            if args.method.lower() in ["snapkv", "h2o"]:
                window_sizes = 32
                kernel_sizes = 7
                pooling = "maxpool"

                layers = len(model.model.layers)
                if not isinstance(window_sizes, list):
                    window_sizes = [window_sizes] * layers
                if not isinstance(max_capacity_prompts, list):
                    max_capacity_prompts = [max_capacity_prompts] * layers
                if not isinstance(kernel_sizes, list):
                    kernel_sizes = [kernel_sizes] * layers
                    
                for i in range(layers):
                    model.model.layers[i].self_attn.config.window_size = window_sizes[i]
                    model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts[i]
                    model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
                    model.model.layers[i].self_attn.config.pooling = pooling
            
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
            
            # --- Generation Logic ---
            if args.method.lower() == "infllm":
                searcher = GreedySearch(model, tokenizer)
                extra_end_token_ids = []
                if args.dataset == "samsum":
                    extra_end_token_ids.append(tokenizer.encode("\n", add_special_tokens=False)[-1])
                
                output = searcher.generate(
                    input_ids=batch_input_ids[0],
                    max_length=args.max_new_tokens or dataset2maxlen[args.dataset],
                    chunk_size=args.chunk_size,
                    extra_end_token_ids=extra_end_token_ids
                )
                
                pred = post_process_infllm(output[0], args.conv_type, args.dataset)
                batch_generations = [pred]
                searcher.clear()

            elif args.method.lower() == "quest":
                with torch.no_grad():
                    output = model(
                        input_ids=tokenized_prompts.input_ids,
                        past_key_values=None,
                        use_cache=True,
                    )
                    past_key_values = output.past_key_values
                    
                    pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    generated_content = [pred_token_idx.item()]
                    
                    for _ in range(output_max_len - 1):
                        outputs = model(
                            input_ids=pred_token_idx,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                        past_key_values = outputs.past_key_values
                        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                        generated_content += [pred_token_idx.item()]
                        if pred_token_idx.item() == tokenizer.eos_token_id:
                            break
                    
                    quest_output_ids = tokenized_prompts.input_ids[0].tolist() + generated_content
                    output = torch.tensor([quest_output_ids], device=tokenized_prompts.input_ids.device)

                    batch_outputs = tokenizer.batch_decode([output[0][context_length:]], skip_special_tokens=True)
                    batch_generations = batch_outputs

            else: # Default generation for FullKV, SentenceKV, etc.
                output = model.generate(
                    **tokenized_prompts,
                    max_new_tokens=output_max_len,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    eos_token_id=[tokenizer.eos_token_id]
                )
                batch_generations = tokenizer.batch_decode([output[0][context_length:]], skip_special_tokens=True)

            torch.cuda.empty_cache()

            for j in range(len(batch_generations)):
                example = {
                    "answers": batch_answerss[j],
                    "pred": batch_generations[j],
                    "length": batch_lengths[j],
                    "dataset": batch_datasets[j],
                    "language": batch_languages[j],
                    "all_classes": batch_all_classess[j],
                    "_id": batch__ids[j]
                }
                fout.write(json.dumps(example) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # General arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="results_long_bench")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True)
    parser.add_argument("--max_num_examples", type=int, default=None)
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"])
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="multifieldqa_en", choices=datasets)
    
    # Method-specific arguments
    parser.add_argument("--method", type=str, default="SentenceKV", 
                        choices=["SentenceKV", "FullKV", "SnapKV", "H2O", "quest", "infllm"])
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", 
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--max_capacity_prompts", type=int, default=1024)
    parser.add_argument("--max_capacity_prompts_ratio", type=float, default=-1)
    parser.add_argument("--semantic_factor", type=float, default=3, help="For SentenceKV")
    parser.add_argument("--chunk_size", type=int, default=32, help="For InfLLM and others")

    # InfLLM specific arguments
    parser.add_argument("--config_path", type=str, default="models/inf_llm/config", help="Path to InfLLM YAML config file")
    parser.add_argument("--conv_type", type=str, default="llama-3.1-inf-llm.yaml", help="Conversation type for InfLLM")


    # Quest specific arguments
    parser.add_argument("--token_budget", type=int, default=None)


    args = parser.parse_args()
    args.save_dir = os.path.join(os.path.dirname(__file__), args.save_dir)
    args.token_budget = args.max_capacity_prompts

    # Configure path to InfLLM config file
    if args.method.lower() == "infllm":
        args.config_path = os.path.join(os.path.dirname(__file__), "..", args.config_path, args.conv_type)

    set_seed(args.seed)
    
    model, tokenizer = None, None
    
    # --- Model & Tokenizer Loading based on Method ---
    print(f"Loading model and tokenizer for method: {args.method}")
    
    if args.method.lower() == "infllm":
        if not args.config_path:
            raise ValueError("--config_path is required when using --method infllm")
        conf = OmegaConf.load(args.config_path)
        if not hasattr(conf.model, "tokenizer_path"):
            conf.model.tokenizer_path = conf.model.path
        model, tokenizer = get_model_and_tokenizer_infllm(conf.model, args)
    else:
        # Default tokenizer loading for other methods
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            use_fast=args.use_fast_tokenizer,
            padding_side="left"
        )
        
        # Model loading for other methods
        if args.method.lower() in ["snapkv", "h2o"]:
            replace_llama(args.method.lower())
            replace_mistral(args.method.lower())    
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, torch_dtype=torch.float16, device_map="auto",
                attn_implementation=args.attn_implementation,
                cache_dir=os.path.join(os.path.expanduser("~"), "huggingface")
            )
        elif args.method.lower() == "fullkv":
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, torch_dtype=torch.float16, device_map="auto",
                attn_implementation=args.attn_implementation,
                cache_dir=os.path.join(os.path.expanduser("~"), "huggingface")
            )
        elif args.method.lower() == "sentencekv":
            config = AutoConfig.from_pretrained(args.model_path)
            from models.sentencekv.custom_llama_sentencekv import LlamaForCausalLM
            # from custom_llama_compress_0928 import LlamaForCausalLM
            punctuations = ['.','!','?','. ','! ','? ','...','\n\n','\n\n\n', '。','！','？','。。']
            punctuations_ids = [tokenizer.convert_tokens_to_ids(p) for p in punctuations]
            config.punctuations_ids = punctuations_ids
            model = LlamaForCausalLM.from_pretrained(
                args.model_path, torch_dtype=torch.float16, device_map="auto", config=config,
                attn_implementation=args.attn_implementation,
                cache_dir=os.path.join(os.path.expanduser("~"), "huggingface")
            )
        elif args.method.lower() == "quest":
            if "llama" in args.model_path.lower() or "longchat" in args.model_path.lower():
                enable_tuple_kv_cache_for_llama()
            if "mistral" in args.model_path.lower():
                enable_tuple_kv_cache_for_mistral()
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=os.path.join(os.path.expanduser("~"), "huggingface")
            )
            enable_quest_attention_eval(model, args)

    # Configure tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.eval()
    
    # Set data file path
    base_data_dir = os.path.join(os.path.dirname(__file__), "data")
    args.data_file = os.path.join(base_data_dir, f"{args.dataset}.jsonl")
    
    print(f"Working on dataset '{args.dataset}' with capacity {args.max_capacity_prompts}")
    
    main(args, model, tokenizer)