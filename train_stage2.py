import os
import sys
import glob
import torch

# Configuration
MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct" 
OUTPUT_ROOT = "output_swift"
STAGE1_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "stage1")
STAGE2_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "stage2")

# Auto-detect cache directory (Copy from train_swift.py)
if os.path.exists("/root/autodl-tmp/models"):
    CACHE_DIR = "/root/autodl-tmp/models"
elif os.path.exists("/root/autodl-tmp"):
    CACHE_DIR = "/root/autodl-tmp/models"
else:
    CACHE_DIR = "./models"

# Env fix
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

def resolve_model_path(cache_dir, model_id_str):
    """
    Find the actual model path
    """
    path1 = os.path.join(cache_dir, model_id_str)
    if os.path.exists(path1):
        return path1
    mangled_model_id = model_id_str.replace("2.5", "2___5")
    path2 = os.path.join(cache_dir, mangled_model_id)
    if os.path.exists(path2):
        return path2
    org_name = model_id_str.split('/')[0]
    org_dir = os.path.join(cache_dir, org_name)
    if os.path.exists(org_dir):
        for folder_name in os.listdir(org_dir):
            if "32B" in folder_name and "Coder" in folder_name:
                return os.path.join(org_dir, folder_name)
    return path1

MODEL_PATH = resolve_model_path(CACHE_DIR, MODEL_ID)

def get_latest_checkpoint(search_dir):
    print(f"Searching for checkpoints in {search_dir}...")
    ckpts = glob.glob(os.path.join(search_dir, "**", "checkpoint-*"), recursive=True)
    ckpts = [c for c in ckpts if os.path.isdir(c)]
    if not ckpts:
        return None
    ckpts.sort(key=os.path.getmtime, reverse=True)
    return ckpts[0]

def run_command(cmd):
    print(f"\nExecuting: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"Command failed with code {ret}")

def run_stage2():
    print(f"\n{'='*20} Starting Stage 2: Fine Optimization (train_data2) {'='*20}")
    
    # 1. Find Stage 1 Checkpoint
    stage1_ckpt = get_latest_checkpoint(STAGE1_OUTPUT_DIR)
    if not stage1_ckpt:
        print(f"Error: No checkpoint found in {STAGE1_OUTPUT_DIR}. Cannot start Stage 2.")
        sys.exit(1)
    
    print(f"Found Stage 1 checkpoint: {stage1_ckpt}")
    
    # 2. Configure paths
    # IMPORTANT: We use the ORIGINAL model path as base, and RESUME from the Stage 1 checkpoint.
    # This loads the base weights + Stage 1 LoRA weights + Optimizer state.
    
    target_modules_str = "q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"

    # 4-GPU Configuration for Stage 2
    cmd = (
        f"CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft "
        f"--model \"{MODEL_PATH}\" "
        f"--resume_from_checkpoint \"{stage1_ckpt}\" "
        f"--dataset train_cypher_all.json "
        f"--output_dir \"{STAGE2_OUTPUT_DIR}\" "
        f"--num_train_epochs 3 "
        f"--max_length 512 "
        f"--per_device_train_batch_size 1 "
        f"--gradient_accumulation_steps 8 "
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
    
    print("Launching Stage 2...")
    run_command(cmd)

if __name__ == "__main__":
    run_stage2()
