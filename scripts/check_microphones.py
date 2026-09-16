import pyaudio

p = pyaudio.PyAudio()

print("--- INPUT DEVICES ---")

for i in range(p.get_device_count()):
    device = p.get_device_info_by_index(i)

    if device["maxInputChannels"] > 0:
        print(
            f"{i}: {device['name']} "
            f"| channels={device['maxInputChannels']} "
            f"| rate={device['defaultSampleRate']}"
        )

p.terminate()