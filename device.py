import sounddevice as sd, json
devs = sd.query_devices()
for i,d in enumerate(devs):
    print(i, repr(d['name']), "in:", d['max_input_channels'], "out:", d['max_output_channels'], "sr:", d.get('default_samplerate'))
