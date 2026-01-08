import os
import sys
import glob
import torch

# Try to import Swift components for Python API execution
try:
    from swift.llm import SftArguments
    from swift.llm.train.sft import SwiftSft
except ImportError:
    print("WARNING: Could not import SftArguments or SwiftSft. Will fall back to CLI if possible, but complex args might fail.")

# Configuration
# Path will be resolved dynamically after download
# Corrected Model ID
MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct" 

OUTPUT_ROOT = "output_swift"

# Auto-detect cache directory
if os.path.exists("/root/autodl-tmp/models"):
    CACHE_DIR = "/root/autodl-tmp/models"
elif os.path.exists("/root/autodl-tmp"):
    CACHE_DIR = "/root/autodl-tmp/models"
else:
    CACHE_DIR = "./models"

# Env fix for RTX 5090 OOM fragmentation
# Note: PYTORCH_CUDA_ALLOC_CONF is recommended by the OOM error message
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

print(f"Using model cache directory: {CACHE_DIR}")

# Construct absolute path to the model to avoid ms-swift assertion errors
def resolve_model_path(cache_dir, model_id_str):
    """
    Find the actual model path, handling ModelScope naming quirks (e.g. 2.5 -> 2___5)
    """
    # 1. Check exact match
    path1 = os.path.join(cache_dir, model_id_str)
    if os.path.exists(path1):
        return path1
    
    # 2. Check name mangling (e.g. 2.5 -> 2___5) which appears in your screenshot
    # Qwen/Qwen2.5-Coder... -> Qwen/Qwen2___5-Coder...
    # Exact match for the screenshot folder structure:
    mangled_model_id = model_id_str.replace("2.5", "2___5")
    path2 = os.path.join(cache_dir, mangled_model_id)
    if os.path.exists(path2):
        print(f"Found model with mangled name at: {path2}")
        # Verify it's not empty (user screenshot showed it empty previously)
        if not os.listdir(path2):
             print(f"WARNING: Found folder at {path2} but it is EMPTY.")
        else:
             return path2

    # 3. Fuzzy search in the organization folder
    org_name = model_id_str.split('/')[0] # e.g. Qwen
    org_dir = os.path.join(cache_dir, org_name)
    if os.path.exists(org_dir):
        for folder_name in os.listdir(org_dir):
            # Look for 32B and Coder
            if "32B" in folder_name and "Coder" in folder_name:
                found_path = os.path.join(org_dir, folder_name)
                print(f"Found model via fuzzy search: {found_path}")
                return found_path

    # Default to original path if nothing found
    return path1

MODEL_PATH = resolve_model_path(CACHE_DIR, MODEL_ID)

def run_command(cmd):
    print(f"\nExecuting: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"Command failed with code {ret}")

# Helper to dynamically resolve argument names for ms-swift/HF
def get_arg_flag(candidates, default):
    try:
        from swift.llm import SftArguments
        from dataclasses import fields
        
        # Get all field names
        field_names = {f.name for f in fields(SftArguments)}
        
        for cand in candidates:
            if cand in field_names:
                return f"--{cand.replace('_', '-')}"
    except ImportError:
        pass
    except Exception as e:
        print(f"Arg resolution warning: {e}")

    return default

def get_latest_checkpoint(search_dir):
    print(f"Searching for checkpoints in {search_dir}...")
    # Find all checkpoint- directories
    ckpts = glob.glob(os.path.join(search_dir, "**", "checkpoint-*"), recursive=True)
    # Filter out invalid ones if any
    ckpts = [c for c in ckpts if os.path.isdir(c)]
    
    if not ckpts:
        return None
    
    # Sort by modification time (latest first)
    ckpts.sort(key=os.path.getmtime, reverse=True)
    return ckpts[0]

def run_stage1():
    print(f"\n{'='*20} Starting Stage 1: Task Understanding (train_data1) {'='*20}")
    output_dir = os.path.join(OUTPUT_ROOT, "stage1")
    
    # Check if model exists at predicted path
    if os.path.exists(MODEL_PATH):
        print(f"Loading local model from: {MODEL_PATH}")
        effective_model_path = MODEL_PATH
    else:
        print(f"Warning: Model not found at {MODEL_PATH}.")
        print(f"Using Model ID '{MODEL_ID}' directly. We expect it to be downloaded to data disk via 'run_swift.sh' configuration.")
        effective_model_path = MODEL_ID

    # Stage 1: Large scale, noisy data (train_cypher_all.json)
    # Reverting to v3.0 standard YAML/JSON config injection workaround or minimal args
    # It appears v3.11+ has issues showing CLI args directly without proper context.
    # However, common HF args usually work if passed positions or strictly.
    # But ValueError: remaining_argv means they were NOT recognized.
    
    # CRITICAL FIX: The issue might be that ms-swift v3 NO LONGER accepts these as CLI args directly
    # for certain backends, OR the parameter names are completely different.
    # We will try the most robust way: Generate a JSON config file and pass it.
    
    config_stage1 = {
        "model": effective_model_path,
        "train_dataset_sample": -1, 
        "dataset": ["train_cypher_all.json"],
        "output_dir": output_dir,
        "num_train_epochs": 1,
        "max_length": 1024,
        "check_dataset_strategy": "none",
        "lora_rank": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": "ALL",
        "gradient_checkpointing": True,
        "batch_size": 1,
        "weight_decay": 0.1,
        "learning_rate": 2e-4,
        "gradient_accumulation_steps": 16,
        "max_grad_norm": 0.5,
        "warmup_ratio": 0.03,
        "evaluation_strategy": "no",
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 2,
        "logging_steps": 10,
        "quantization_bit": 4, # Trying old name in JSON, sometimes parser handles json differently
        "lazy_tokenize": True
    }
    
    import json
    json_path = os.path.join(OUTPUT_ROOT, "stage1_args.json")
    with open(json_path, 'w') as f:
        json.dump(config_stage1, f)
        
    # We use a trick: If CLI args fail, passing a config file is often safer
    # But ms-swift might not support --config directly in all versions.
    # Let's try the Python API again but with a specific import fix
    # IF CLI fails, we must rely on Python API.
    # The previous import error suggests `swift.llm` structure changed.
    
    # 2nd Attempt: CLI with known good generic args, dropping specific ones that fail
    # We drop 'sft_type' (implied by lora args usually)
    # We drop 'quantization' args if they keep failing, and rely on model loading to handle it
    # BUT we need quantization for 32B.
    
    # Let's try "bnb_4bit_quant_type": "nf4" style if possible
    # Or just use the SWIFT_UI_PARAMS environment variable trick? No.
    
    # FORCE USE OF Python API which we can debug better than opaque CLI
    # We need to find where SftArguments is.
    # Based on Swift 3.x, it's often in swift.llm.sft
    
    # Determine correct arguments dynamically
    arg_batch = get_arg_flag(['per_device_train_batch_size', 'batch_size'], '--per_device_train_batch_size')
    arg_lora = get_arg_flag(['lora_target_modules', 'target_modules'], '--target_modules')
    arg_quant = get_arg_flag(['quant_bits', 'quantization_bit', 'bits'], '--quant_bits')
    
    print(f"Resolved Args: Batch={arg_batch}, LoRA={arg_lora}, Quant={arg_quant}")

    # FORCE device_map auto to handle large shard (16GB+) loading on 32GB GPU
    # disable device_map="auto" to use device_map="cuda:0" explicitly, but we need simple args for CLI
    # The error 'Tensor on device meta is not on the expected device cuda:0!' with LoRA usually comes 
    # from accelerate/bitsandbytes conflict when using device_map='auto' with qlora on single GPU.
    # It suggests some layers stayed on 'meta' device or CPU offloaded and didn't move back correctly.
    
    # We will REMOVE --device_map auto and strictly trust the trainer to handle single GPU.
    # We rely on low_cpu_mem_usage (implicit in swift) and quantization to fit.
    
    # Stage 1: Configure specific target modules
    target_modules_str = "q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"

    cmd = (
        f"CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft "
        f"--model \"{effective_model_path}\" "
        f"--dataset train_cypher_all.json "
        f"--output_dir \"{output_dir}\" "
        f"--num_train_epochs 1 "
        f"--max_length 512 "
        f"--per_device_train_batch_size 1 "
        f"--gradient_accumulation_steps 4 "
        f"--learning_rate 2e-4 "
        f"--target_modules {target_modules_str} "
        f"--quant_bits 4 " 
        f"--gradient_checkpointing true "
        f"--save_total_limit 2 "
        f"--max_grad_norm 0.5 "
        f"--warmup_ratio 0.03 "
        f"--eval_steps 500 " 
        f"--save_steps 500 "
        f"--logging_steps 10 "
        f"--bnb_4bit_compute_dtype bfloat16 "
        f"--bnb_4bit_quant_type nf4 "
        f"--bnb_4bit_use_double_quant true"
    )
    # NOTE: If this fails again with 'remaining_argv', it means swift sft is STRICTLY ignoring these.
    # There is a high chance only 'model', 'dataset' etc are parsed generally, and others need 
    # specific flags or are not available in the lightweight sft entry point?
    # NO, sft.py usually handles all.
    
    # Let's try the SUPER COMPATIBLE set (v3.0 often re-added these aliases)
    run_command(cmd)
    
    # Return the best checkpoint path for Stage 2
    return get_latest_checkpoint(output_dir)

def run_stage2(stage1_checkpoint):
    print(f"\n{'='*20} Starting Stage 2: Fine Optimization (train_data2) {'='*20}")
    output_dir = os.path.join(OUTPUT_ROOT, "stage2")
    
    print(f"Starting Stage 2 from Stage 1 checkpoint: {stage1_checkpoint}")

    # Stage 2: High quality, small scale (train_cypher_v2.json)
    
    # Determine correct arguments dynamically (re-run as it's cheap)
    arg_batch = get_arg_flag(['per_device_train_batch_size', 'batch_size'], '--per_device_train_batch_size')
    arg_lora = get_arg_flag(['lora_target_modules', 'target_modules'], '--target_modules')
    arg_quant = get_arg_flag(['quant_bits', 'quantization_bit', 'bits'], '--quant_bits')

    # Stage 2: Configure specific target modules
    target_modules_str = "q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"

    cmd = (
        f"CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft "
        f"--model \"{stage1_checkpoint}\" "
        f"--dataset train_cypher_all.json "
        f"--output_dir \"{output_dir}\" "
        f"--num_train_epochs 3 "
        f"--max_length 512 "
        f"--per_device_train_batch_size 1 "
        f"--gradient_accumulation_steps 8 "
        f"--learning_rate 2e-4 "
        f"--target_modules {target_modules_str} "
        f"--quant_bits 4 " 
        f"--gradient_checkpointing true "
        f"--save_total_limit 2 "
        f"--bnb_4bit_compute_dtype bfloat16 "
        f"--bnb_4bit_quant_type nf4 "
        f"--bnb_4bit_use_double_quant true"
    )
    
    run_command(cmd)

if __name__ == "__main__":
    # Ensure output root exists
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    try:
        # Run Stage 1
        best_ckpt_stage1 = run_stage1()
        
        if best_ckpt_stage1 and os.path.exists(best_ckpt_stage1):
            print(f"Stage 1 completed. Found checkpoint: {best_ckpt_stage1}")
            print(f"To run Stage 2, please execute: python train_stage2.py")
        else:
            print("Stage 1 did not return a valid checkpoint. Exiting.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)
