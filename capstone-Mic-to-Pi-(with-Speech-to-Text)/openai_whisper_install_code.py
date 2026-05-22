# Import from Hugging Face
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


# ====== User Model Size Config ======
# 0=tiny.en, 1=base.en, 2=large
downloadSize = 0
# =========================

# Model mapping
model_map = {
    0: ("openai/whisper-tiny.en", "whisper-tiny.en"),
    1: ("openai/whisper-base.en", "whisper-base.en"),
    2: ("openai/whisper-large-v3", "whisper-large-v3")
}

# Validate input
if downloadSize not in model_map:
    raise ValueError("Use 0 (tiny), 1 (base), or 2 (large) as downloadSize other than: " + str(downloadSize))

repo_id, folder_name = model_map[downloadSize]
save_dir = f"./{folder_name}"
size_name = folder_name

# ===== BEFORE DOWNLOAD =====
print(f"The {size_name} model of speech-to-text is downloading...")

# Download model + processor
model = AutoModelForSpeechSeq2Seq.from_pretrained(repo_id)
processor = AutoProcessor.from_pretrained(repo_id)

# Local Save
model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)

# ===== AFTER DOWNLOAD =====
print(f"The {size_name} model is installed successfully.")


"""
To Uninstall:
In order to uninstall, just delete the model file directly
"""