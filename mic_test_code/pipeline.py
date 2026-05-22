import os
import sounddevice as sd
import soundfile as sf
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv() 
client = OpenAI()

sd.default.device = "hw:2,0"

sample_rate = 48000
channels = 2
duration = 5
audio_file = "temp_capture.wav"
output_text_file = "transcription.txt"

def main():
    print(f"🎤 Recording started ({duration} seconds)...")
    
    # Capture audio using your existing I2S parameters
    audio_data = sd.rec(
        int(sample_rate * duration),
        samplerate=sample_rate,
        channels=channels,
        dtype='int32'
    )
    sd.wait()
    sf.write(audio_file, audio_data, sample_rate)
    print("✅ Recording complete. Sending to OpenAI API for transcription...")

    # Send the audio file to the OpenAI Whisper API
    try:
        with open(audio_file, "rb") as audio_file_obj:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file_obj,
                language="en" # Matches your previous language kwarg
            )
        
        result_text = transcription.text.strip()
        
        # Save to text file
        with open(output_text_file, "w", encoding="utf-8") as f:
            f.write(result_text)

        print("\n📝 Transcription Result:")
        print(result_text)

    except Exception as e:
        print(f"\n❌ Error during API call: {e}")

if __name__ == "__main__":
    main()