import os
import shutil
from modelscope.hub.snapshot_download import snapshot_download

# Configuration for download
# Corrected Model IDs for ModelScope
MODELS = {
    "coder_32b": "Qwen/Qwen2.5-Coder-32B-Instruct", # Corrected from 32B-Coder
    "instruct_14b": "Qwen/Qwen2.5-14B-Instruct"
}

# AutoDL typically uses /root/autodl-tmp for large data storage
if os.path.exists("/root/autodl-tmp"):
    BASE_DIR = "/root/autodl-tmp"
    print(f"SUCCESS: Detected AutoDL data disk at {BASE_DIR}. Models will be stored here.")
else:
    print("WARNING: /root/autodl-tmp NOT FOUND!")
    print("Using current directory (system disk). This may fail due to lack of space for 32B models.")
    BASE_DIR = "."

DOWNLOAD_DIR = os.path.join(BASE_DIR, "models")

def move_existing_models():
    """Check if models exist in the wrong place (system disk) and move them."""
    old_dir = "./models"
    if os.path.abspath(old_dir) != os.path.abspath(DOWNLOAD_DIR) and os.path.exists(old_dir):
        print(f"Detected 'models' in system disk ({old_dir}). Moving to data disk ({DOWNLOAD_DIR})...")
        if not os.path.exists(DOWNLOAD_DIR):
            os.makedirs(DOWNLOAD_DIR)
        
        # We use os.system for moving to be robust across filesystems
        # 'cp -r' then 'rm -rf' is safer than mv for cross-device
        ret = os.system(f"cp -r -n {old_dir}/* {DOWNLOAD_DIR}/")
        if ret == 0:
            print("Move successful. freeing up space on system disk...")
            os.system(f"rm -rf {old_dir}")
        else:
            print("Warning: Failed to move models automatically. Please check disk space.")

def download_models():
    # Attempt to free space first
    move_existing_models()

    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    
    print(f"Downloading models to {os.path.abspath(DOWNLOAD_DIR)}...")

    # 1. Download Qwen2.5-32B-Coder-Instruct (For Training & Main Inference)
    print(f"\n>>> Downloading {MODELS['coder_32b']}...")
    
    # Pre-check for empty/corrupt directories to force fresh download
    expected_path_part = MODELS['coder_32b'].replace("2.5", "2___5") # Handle mangled name
    possible_dir = os.path.join(DOWNLOAD_DIR, expected_path_part)
    if os.path.exists(possible_dir):
        if not os.listdir(possible_dir):
            print(f"Found empty directory at {possible_dir}. Deleting to allow fresh download.")
            try:
                shutil.rmtree(possible_dir)
            except:
                pass
    
    try:
        model_dir_32b = snapshot_download(MODELS['coder_32b'], cache_dir=DOWNLOAD_DIR)
        print(f"Successfully downloaded 32B Coder to: {model_dir_32b}")
    except Exception as e:
        print(f"Failed to download 32B Coder: {e}")

    # 2. Download Qwen2.5-14B-Instruct (Optional for now, preventing full disk)
    # Only download if we are sure we have space or if the user specifically requests it.
    # For this training task, 32B is the priority.
    print(f"\n>>> Downloading {MODELS['instruct_14b']} (Skipping if disk is full)...")
    try:
        # Check free space (very rough check)
        stat = os.statvfs(DOWNLOAD_DIR)
        free_space_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        
        if free_space_gb > 30: # 14B model needs ~30GB
            model_dir_14b = snapshot_download(MODELS['instruct_14b'], cache_dir=DOWNLOAD_DIR)
            print(f"Successfully downloaded 14B Instruct to: {model_dir_14b}")
        else:
            print(f"Skipping 14B model download. Remaining space ({free_space_gb:.2f} GB) might be insufficient.")
    except Exception as e:
        print(f"Failed/Skipped download 14B Instruct: {e}")

    print("\nDownload process finished.")

if __name__ == "__main__":
    download_models()
