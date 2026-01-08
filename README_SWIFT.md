# ms-swift Training Instructions

Since you are using `ms-swift` on the server (AutoDL), I have created specific scripts to leverage its powerful CLI and Trainer logic.

## 1. Installation
On your server terminal:
```bash
pip install 'ms-swift[llm]' -U
```

## 2. Training (Two-Stage)
Run the following script which handles both Stage 1 (Task Understanding) and Stage 2 (Fine Optimization):

```bash
bash run_swift.sh
```
Or directly via Python:
```bash
python train_swift.py
```

## 3. Configuration
The training configuration is defined in `train_swift.py`.
- **Model Path**: Defaults to `./Qwen/Qwen2___5-3B`. Change `MODEL_PATH` variable if you use the 32B model.
- **Quantization**: Defaults to `4-bit` (QLoRA) which is optimal for memory efficiency.
- **Batch Size**: Configured for 2-phase training.

## 4. Inference
After training, you can use `ms-swift` for inference:
```bash
swift infer --ckpt_dir output_swift/stage2/checkpoint-xxx --load_dataset test_normal --eval_human true
```
Or use the provided `inference.py` (ensure you point it to the new `output_swift` directory).
