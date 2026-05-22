import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 48000
CHANNELS = 2
DURATION = 5
OUTPUT_FILE = "voice_command.wav"

# Explicitly set the input device to your I2S mic (Card 2)
sd.default.device = "hw:2,0"

def record_audio():
    print(f"🎤 Recording started ({DURATION} seconds)...")
    
    audio_data = sd.rec(
        int(SAMPLE_RATE * DURATION),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='int32'
    )
    
    sd.wait() 
    print("✅ Recording complete.")
    
    sf.write(OUTPUT_FILE, audio_data, SAMPLE_RATE)
    print(f"💾 Audio saved successfully to {OUTPUT_FILE}")

if __name__ == "__main__":
    record_audio()