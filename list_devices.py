import sounddevice as sd

print("Available audio devices:")
print(sd.query_devices())

default_input = sd.default.device[0]
default_output = sd.default.device[1]

print(f"\nDefault Input Device ID: {default_input}")
print(f"Default Output Device ID: {default_output}")

# Try to find a 'monitor' device automatically
devices = sd.query_devices()
monitor_id = None
for i, dev in enumerate(devices):
    if 'monitor' in dev['name'].lower():
        print(f"Potential system audio monitor found: ID {i} ({dev['name']})")
        monitor_id = i

if monitor_id is None:
    print("No device with 'monitor' in name found. You might need to set it manually.")
else:
    print(f"Suggested device ID for backend.py: {monitor_id}")
