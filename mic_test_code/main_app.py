import os
import time
import sounddevice as sd
import soundfile as sf
from openai import OpenAI
from dotenv import load_dotenv
from openwakeword.model import Model
from pathlib import Path

# --- Configuration & Initialization ---
load_dotenv() 
client = OpenAI()

# Force sounddevice to use your specific I2S hardware mic setup
sd.default.device = "hw:2,0"

# Audio & Transcription Settings
SAMPLE_RATE = 48000
RECORDING_DURATION = 5
AUDIO_FILE = "temp_capture.wav"
OUTPUT_TEXT_FILE = "transcription.txt"

# Load Wake Word Model (Using Option 1 fix for older API compatibility)
DIR = str(Path(__file__).resolve().parent / "SecondIteration" / "sherlock.onnx")
_oww_model = Model(wakeword_model_paths=[DIR])

def wakeword_listener(channels=2):
    """Listens continuously until 'sherlock' is detected, then returns True."""
    wake_word_detected = False
    
    def audio_callback(indata, frames, time, status):
        nonlocal wake_word_detected
        if status:
            print(status, flush=True)
            
        # Downsample from 48k to 16k and grab the first channel for openwakeword
        audio_16k_mono = indata[::3, 0]
        prediction = _oww_model.predict(audio_16k_mono)
        
        # Threshold for detection
        if prediction.get('sherlock', 0) > 0.5:
            wake_word_detected = True
            _oww_model.reset()

    # Open stream. The 'with' block ensures the mic is released once detection occurs!
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=channels, blocksize=3840, 
                        dtype='int16', callback=audio_callback):
        while not wake_word_detected:
            sd.sleep(100)
            
    return True

def record_and_transcribe(channels=2):
    """Records 5 seconds of audio and sends it to OpenAI Whisper."""
    print(f"\n🎤 Recording started ({RECORDING_DURATION} seconds)...")
    
    # Capture audio using your existing I2S parameters
    audio_data = sd.rec(
        int(SAMPLE_RATE * RECORDING_DURATION),
        samplerate=SAMPLE_RATE,
        channels=channels,
        dtype='int32'
    )
    sd.wait() # Block execution until the recording finishes
    sf.write(AUDIO_FILE, audio_data, SAMPLE_RATE)
    print("✅ Recording complete. Sending to OpenAI API for transcription...")

    # Send the audio file to the OpenAI Whisper API
    try:
        with open(AUDIO_FILE, "rb") as audio_file_obj:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file_obj,
                language="en"
            )
        
        result_text = transcription.text.strip()
        
        # Save to text file
        with open(OUTPUT_TEXT_FILE, "w", encoding="utf-8") as f:
            f.write(result_text)

        print("\n📝 Transcription Result:")
        print("-" * 30)
        print(result_text)
        print("-" * 30)

    except Exception as e:
        print(f"\n❌ Error during API call: {e}")

def text_to_command(filePath : str):

    with open(OUTPUT_TEXT_FILE, "r", encoding="utf-8") as f:
            translated_text = f.read()

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
        command_type = "CHECK_WATER_LEAK" # normal or leaking

    # Motion of Light
    elif detected_action == "check" and detected_target == "motion":
        command_type = "CHECK_MOTION_LIGHT" # on or off
    
    # Status of Camera
    elif detected_action == "check" and detected_target == "camera":
        command_type = "CHECK_CAMERA_STATUS" # on or off
    
    else:
        command_type = "UNKNOWN_COMMAND"

    print("\n🔎 Keyword Catch Result:")
    print(f"Original text: {translated_text}")
    print(f"Detected action group: {detected_action}")
    print(f"Detected target group: {detected_target}")
    print(f"Command type received: {command_type}")

    return command_type

    

def main():
    # Attempt to query the max channels of the specific hw:2,0 device
    default_input = sd.default.device[0]
    try:
        device_info = sd.query_devices(default_input, 'input')
        channels = int(device_info['max_input_channels'])
    except Exception as e:
        print(f"⚠️ Could not query max channels. Defaulting to 2. Error: {e}")
        channels = 2

    print("🚀 System ready and initialized.")
    
    # Infinite loop to keep the assistant alive
    while True:
        print("\n👂 Listening for wake word 'Sherlock'...")
        
        # This will block until the wake word is spoken
        if wakeword_listener(channels=channels):
            print("🔔 Wake word detected!")
            
            # Record the follow-up command and transcribe
            record_and_transcribe(channels=channels)

            # Transcribe to Command
            text_to_command(OUTPUT_TEXT_FILE)
            
            print("\n⏳ Resuming wake word detection in 2 seconds...")
            time.sleep(2) 

if __name__ == "__main__":
    main()