import torch
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
try:
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Capability: {torch.cuda.get_device_capability(0)}")
        
        # Memory Check
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print(f"Total Memory: {t / 1024**3:.2f} GB")
        print(f"Reserved Memory: {r / 1024**3:.2f} GB")
        print(f"Allocated Memory: {a / 1024**3:.2f} GB")
        
        # Allocation Test
        try:
            print("Attempting 1GB allocation...")
            x = torch.ones(256 * 1024 * 1024, dtype=torch.float32, device="cuda:0")
            print("Allocation successful.")
            del x
        except Exception as e:
            print(f"Allocation failed: {e}")

        # BitsAndBytes Check
        try:
            import bitsandbytes as bnb
            print(f"BitsAndBytes Version: {bnb.__version__}")
            print("BitsAndBytes imported successfully.")
        except ImportError:
            print("BitsAndBytes NOT installed.")
        except Exception as e:
            print(f"BitsAndBytes import error: {e}")

except Exception as e:
    print(f"CUDA Error: {e}")
