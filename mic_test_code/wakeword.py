import sounddevice as sd
import openwakeword
from openwakeword.model import Model
from pathlib import Path
import time

#Downloads Spectrogram and other required models, uncomment line below if getting error
#openwakeword.utils.download_models()

#Edit to match actual model location and filetype (whether tflite or onnx)
DIR = str(Path(__file__).resolve().parent / "SecondIteration" / "sherlock.onnx")
<<<<<<< HEAD
#_oww_model = Model(wakeword_models=[DIR])
_oww_model = Model(wakeword_model_paths=[DIR])
=======
_oww_model = Model(wakeword_models=[DIR])
>>>>>>> ef7fb7cbbbeef75634e47a470f97e851e4a176c1

#Provide amount of channels. Defaults to 2, as you can see
def wakeword_listener(channels=2):
    wake_word_detected = False
    
    def audio_callback(indata, frames, time, status):
        nonlocal wake_word_detected
        audio_16k_mono = indata[::3, 0]
        prediction = _oww_model.predict(audio_16k_mono)
        if prediction.get('sherlock', 0) > 0.5:
            wake_word_detected = True
            _oww_model.reset()

    with sd.InputStream(samplerate=48000, channels=channels, blocksize=3840, 
                        dtype='int16', callback=audio_callback):
        while not wake_word_detected:
            sd.sleep(100)
            
    return True

if __name__ == "__main__":
    sample_rate = 48000
    #Next 3 lines ensures script won't fail catastrophically if input is only monochannel as it is
    default_input = sd.default.device[0]
    device_info = sd.query_devices(default_input, 'input')
    channels = max_channels = int(device_info['max_input_channels'])
    print("Testing")
    while(True):
        if wakeword_listener(channels=max_channels):
            print("Success")
            time.sleep(1)