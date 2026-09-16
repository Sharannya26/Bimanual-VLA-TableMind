import pyaudio

DEVICE_INDEX = 1
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 4096

p = pyaudio.PyAudio()

print("Opening microphone...")
print(f"Device: {DEVICE_INDEX}")
print(f"Sample rate: {SAMPLE_RATE}")
print(f"Channels: {CHANNELS}")

stream = p.open(
    format=pyaudio.paInt16,
    channels=CHANNELS,
    rate=SAMPLE_RATE,
    input=True,
    input_device_index=DEVICE_INDEX,
    frames_per_buffer=CHUNK_SIZE,
)

print()
print("Microphone opened successfully!")
print("Speak for a few seconds...")
print("Press Ctrl+C to stop.")
print()

try:
    for _ in range(50):
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        print(f"Captured {len(data)} bytes")

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Microphone closed.")