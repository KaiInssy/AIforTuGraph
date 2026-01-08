import json
import torch
import os
import glob
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ----------------- CONFIGURATION -----------------
# PLEASE UPDATE THIS PATH TO YOUR LOCAL MODEL (BASE MODEL)
POSSIBLE_PATHS = [
    "/root/autodl-tmp/models/Qwen/Qwen2___5-Coder-32B-Instruct", 
    "/root/autodl-tmp/models/Qwen/Qwen2.5-Coder-32B-Instruct",
    "./models/Qwen/Qwen2___5-Coder-32B-Instruct",
    "./models/Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct" 
]

MODEL_PATH = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        MODEL_PATH = p
        break

if MODEL_PATH is None:
    MODEL_PATH = "Qwen/Qwen2.5-Coder-32B-Instruct" 
    print(f"Warning: Local 32B model not found. Using ID: {MODEL_PATH}")
else:
    print(f"Using detected Base Model path: {MODEL_PATH}")

# STAGE 1 ADAPTER PATH
STAGE1_DIR = "output_swift/stage1"
ADAPTER_PATH = None

if os.path.exists(STAGE1_DIR):
    # Find the latest checkpoint recursively
    print(f"Searching for checkpoints in {STAGE1_DIR}...")
    checkpoints = glob.glob(os.path.join(STAGE1_DIR, "**", "checkpoint-*"), recursive=True)
    # Filter for valid checkpoints (must have adapter_config.json)
    valid_checkpoints = []
    for d in checkpoints:
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "adapter_config.json")):
            valid_checkpoints.append(d)
    
    if valid_checkpoints:
        # Sort by modification time (latest first)
        valid_checkpoints.sort(key=os.path.getmtime, reverse=True)
        ADAPTER_PATH = valid_checkpoints[0]
        print(f"Auto-detected Stage 1 checkpoint: {ADAPTER_PATH}")
    else:
        print(f"Warning: {STAGE1_DIR} exists but contains no valid 'checkpoint-*' folders (with adapter_config.json).")
else:
    # Use Fallback if user used old output directory name
    if os.path.exists("output_cypher_lora"):
        ADAPTER_PATH = "output_cypher_lora"
        print(f"Using fallback Stage 1 path: {ADAPTER_PATH}")

if ADAPTER_PATH is None:
    print(f"ERROR: Could not find any Stage 1 checkpoint in {STAGE1_DIR} or output_cypher_lora")
    # Set a dummy path so it fails gracefully later or user can edit
    ADAPTER_PATH = "output_swift/stage1/checkpoint-xxx"

TEST_FILE = "test_cypher.json"
OUTPUT_FILE = "submit_cypher_stage1.json"
# -------------------------------------------------

def get_instruction(db_id):
    if db_id in ["movie", "common"]:
        return (
            "我希望你像一个Tugraph数据库前端一样工作，你只需要返回给我cypher语句。"
            "下面是一条描述任务的指令，写一条正确的response来完成这个请求.\n\"\n##Instruction:\n"
            "movie包含节点person、genre、keyword、movie、user和边acted_in、rate、directed、"
            "is_friend、has_genre、has_keyword、produce、write。节点person有属性id、name、born、poster_image。"
            "节点genre有属性id、name。节点keyword有属性id、name。节点movie有属性id、title、tagline、summary、"
            "poster_image、duration、rated。节点user有属性id、login。边acted_in有属性role。边rate有属性stars。\n\n"
        )
    else:
        return (
            "我希望你像一个Tugraph数据库前端一样工作，你只需要返回给我cypher语句。"
            "下面是一条描述任务的指令，写一条正确的response来完成这个请求.\n\"\n##Instruction:\n"
            "请根据输入的自然语言生成正确的Cypher查询语句。\n\n"
        )

def post_process_cypher(text):
    text = text.strip()
    # Remove markdown code blocks
    if "```" in text:
        import re
        pattern = r"```(?:cypher)?\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1)
        else:
            text = text.replace('```cypher', '').replace('```', '')
    
    # Remove inline code backticks
    if text.startswith('`') and text.endswith('`') and len(text) > 2:
        text = text[1:-1]

    # Remove "Response:" prefix if present
    if text.lower().startswith("response:"):
        text = text[9:].strip()
        
    # Deduplicate repeated lines (common failure mode)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if len(lines) > 1 and all(line == lines[0] for line in lines):
        text = lines[0]
        
    return text.strip()

def predict():
    if not os.path.exists(TEST_FILE):
        print(f"Error: {TEST_FILE} not found.")
        return

    # Load Model
    print(f"Loading Stage 1 Adapter from: {ADAPTER_PATH}")
    try:
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, 
            device_map="auto", 
            trust_remote_code=True,
            quantization_config=bnb_config
        )
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load Test Data
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test cases.")

    results = []
    
    for i, item in enumerate(test_data):
        db_id = item.get("db_id", "common")
        question = item["question"]
        
        instruction = get_instruction(db_id)
        prompt = f"{instruction}{question}\n\nResponse:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512,
                num_beams=1,
                early_stopping=False,
                do_sample=False, 
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id
            )
        
        output_tokens = outputs[0][len(inputs.input_ids[0]):]
        response_cypher = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
        response_cypher = post_process_cypher(response_cypher)
        
        print(f"[{i+1}/{len(test_data)}] Generated: {response_cypher}")
        
        result_item = item.copy()
        result_item["output"] = response_cypher
        results.append(result_item)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    predict()
