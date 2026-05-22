# Import
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Force offline setting (enable it if you want)
import os
# ==========================================================
# Offline Control Code
# ==========================================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# ==========================================================


# ==========================================================
# Model Selection
# 0 = tiny.en
# 1 = base.en
# 2 = large-v3

downloadSize = 0
# Audio file input
inputAudio = ("your-audio-file")
# ==========================================================

model_map = {
    0: ("tiny.en", "./whisper-tiny.en"),
    1: ("base.en", "./whisper-base.en"),
    2: ("large-v3", "./whisper-large-v3"),
}

if downloadSize not in model_map:
    raise ValueError("Invalid downloadSize. Use 0 (tiny.en), 1 (base.en), or 2 (large-v3).")

model_name, model_path = model_map[downloadSize]
# Check whether model is installed locally
if not os.path.isdir(model_path):
    raise FileNotFoundError(
        f"the model {model_name} is not installed yet, please install before use."
    )


# Model Setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, local_files_only=True,
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_path, local_files_only=True,)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)


result = pipe(
    inputAudio,
    generate_kwargs={
        "language": "english",
        "task": "transcribe"
    }
)

print("\n\nBelow is the output text from speech:")
print(result["text"])