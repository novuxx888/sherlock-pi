import os
import sounddevice as sd
import soundfile as sf
# from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from scipy.signal import resample_poly

# import serial
# import time

load_dotenv() 
# client = OpenAI()

sd.default.device = "hw:2,0"

sample_rate = 48000
target_rate = 16000
channels = 2
duration = 5
audio_file = "temp_capture.wav"
local_audio_file = "temp_capture_16k_mono.wav"
output_text_file = "transcription.txt"
modelVersion = 0 # 0 = local version(tiny), 1 = api version
model_path = "./whisper-tiny.en"

# Safety cleanup switch
auto_delete_audio_files = True


# ESP
esp_serial_port = "/dev/serial0"
esp_baud_rate = 115200


def prepare_audio_for_whisper(input_file, output_file, input_rate=48000, output_rate=16000):
    audio_data, sr = sf.read(input_file, dtype="float32")

    # Convert stereo/multi-channel to mono
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Resample to 16 kHz if needed
    if sr != output_rate:
        gcd = np.gcd(sr, output_rate)
        up = output_rate // gcd
        down = sr // gcd
        audio_data = resample_poly(audio_data, up, down)

    # Avoid clipping
    max_val = np.max(np.abs(audio_data))
    if max_val > 1.0:
        audio_data = audio_data / max_val

    sf.write(output_file, audio_data, output_rate)
    return output_file

def transcribe_local_whisper(audio_path):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Local model folder not found: {model_path}\n"
            "Make sure your Whisper tiny model is installed there."
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"🧠 Loading local Whisper model from: {model_path}")
    print(f"🖥️ Device: {device}")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )

    asr_pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=0 if device.startswith("cuda") else -1,
    )

    result = asr_pipe(
        audio_path,
        generate_kwargs={

        },
    )

    return result["text"].strip()


def transcribe_openai_api(audio_path):
    with open(audio_path, "rb") as audio_file_obj:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file_obj,
            language="en",
        )

    return transcription.text.strip()

def delete_audio_files_from_this_run(files_created_this_run):
    """
    Deletes only audio files created during this script run.
    It does not scan folders or delete older saved files.
    """
    for file_path in files_created_this_run:
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"🗑️ Deleted audio file: {file_path}")
        except Exception as e:
            print(f"⚠️ Could not delete {file_path}: {e}")
            
def wordCatch(translated_text):
    """
    Catches command keywords from transcribed text and identifies
    what type of command was received.

    Supported examples:
    - open light
    - turn on light
    - close window
    - turn off light
    - switch off light
    - check weather
    - evaluate water leak
    """

    text = translated_text.lower().strip()

    # Command: Action
    action_groups = {
        "open_on": ["open", "turn on"],
        "close_off": ["close", "turn off", "switch off"],
        "check": ["check", "evaluate", "find"],
    }

    target_groups = {
        "light": ["light", "lights"],
        # "window": ["window", "windows"],
        "weather": ["weather"], # whole data
        "temperature" : ["temperature"],
        "pressure" : ["pressure, air pressure"],
        "humidity" : ["humidity, how wet"],
        "water_leak": ["water leak", "water leakage", "leak"],
        "motion" : ["motion"],
        "camera" : ["camera"]
    }

    detected_action = None
    detected_target = None

    # Find action keyword
    for action_type, keywords in action_groups.items():
        for keyword in keywords:
            if keyword in text:
                detected_action = action_type
                break
        if detected_action:
            break

    # Find target keyword
    for target_type, keywords in target_groups.items():
        for keyword in keywords:
            if keyword in text:
                detected_target = target_type
                break
        if detected_target:
            break

    # Decide command type
    # Light (Unsured)
    if detected_action == "open_on" and detected_target == "light":
        command_type = "TURN_LIGHT_ON"

    elif detected_action == "close_off" and detected_target == "light":
        command_type = "TURN_LIGHT_OFF"

    # elif detected_action == "open_on" and detected_target == "window":
    #     command_type = "OPEN_WINDOW"

    # elif detected_action == "close_off" and detected_target == "window":
    #     command_type = "CLOSE_WINDOW"

    # BME
    elif detected_action == "check" and detected_target == "weather":
        command_type = "CHECK_WEATHER"

    elif detected_action == "check" and detected_target == "temperature":
        command_type = "CHECK_TEMPERATURE"

    elif detected_action == "check" and detected_target == "pressure":
        command_type = "CHECK_PRESSURE"

    elif detected_action == "check" and detected_target == "humidity":
        command_type = "CHECK_HUMIDITY"

    # Water Leak
    elif detected_action == "check" and detected_target == "water_leak":
        command_type = "CHECK_WATER_LEAK"

    # Motion of Light
    elif detected_action == "check" and detected_target == "motion":
        command_type = "CHECK_MOTION_LIGHT"
    
    # Status of Camera
    elif detected_action == "check" and detected_target == "camera":
        command_type = "CHECK_CAMERA_STATUS"
    
    else:
        command_type = "UNKNOWN_COMMAND"

    print("\n🔎 Keyword Catch Result:")
    print(f"Original text: {translated_text}")
    print(f"Detected action group: {detected_action}")
    print(f"Detected target group: {detected_target}")
    print(f"Command type received: {command_type}")

    return command_type



def main():
    files_created_this_run = []
    try:
        print(f"🎤 Recording started ({duration} seconds)...")

        audio_data = sd.rec(
            int(sample_rate * duration),
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
        )

        sd.wait()

        sf.write(audio_file, audio_data, sample_rate)
        files_created_this_run.append(audio_file)

        print(f"✅ Recording complete: {audio_file}")

        whisper_ready_audio = prepare_audio_for_whisper(
            input_file=audio_file,
            output_file=local_audio_file,
            output_rate=target_rate,
        )
        files_created_this_run.append(local_audio_file)

        if modelVersion == 0:
            print("🧠 Using local Whisper tiny model...")
            result_text = transcribe_local_whisper(whisper_ready_audio)

        elif modelVersion == 1:
            print("☁️ Using OpenAI Whisper API...")
            result_text = transcribe_openai_api(whisper_ready_audio)

        else:
            raise ValueError("modelVersion must be 0 for local or 1 for API.")

        with open(output_text_file, "w", encoding="utf-8") as f:
            f.write(result_text)

        print("\n📝 Transcription Result:")
        print(result_text)
        print(f"\n💾 Saved text to: {output_text_file}")

        command_type = wordCatch(result_text)

        print(f"\n🤖 Final command type: {command_type}")
        print(f"\n💾 Saved text to: {output_text_file}")

        # sendCommandToESP(command_type)

    except Exception as e:
        print(f"\n❌ Error during transcription: {e}")

    finally:
        if auto_delete_audio_files:
            print("\n🧹 Auto-delete enabled. Cleaning audio files from this run...")
            delete_audio_files_from_this_run(files_created_this_run)
        else:
            print("\n📁 Auto-delete disabled. Audio files were kept.")

if __name__ == "__main__":
    main()