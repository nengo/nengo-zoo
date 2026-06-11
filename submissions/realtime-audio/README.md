# realtime-audio

Live microphone audio injected into a spiking population, visualised in NengoGUI.

## Description

A small NengoGUI-first model: a `nengo.Node` reads the most recent ~10 ms of microphone audio (captured via `sounddevice` / PortAudio), resamples it to a fixed `AUDIO_DIMS`-length window, and routes the window into a spiking ensemble. The mic samples are written to a ring buffer in the PortAudio callback thread and the node always returns the *latest* window, so the display stays live whether the simulation runs faster or slower than real time.

Cross-platform — the `sounddevice` pip wheels bundle the PortAudio binaries on macOS and Windows, so no OS-specific setup is needed beyond granting your terminal microphone permission.

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
nengo realtime_audio.py
```

In the browser GUI, right-click the **`audio neurons`** ensemble to add a *value* plot and a *spike raster*, press play, and speak into the mic. The shipped `realtime_audio.py.cfg` already lays out a value plot and raster in the usual spots.

Tweak `AUDIO_DIMS`, `WINDOW_MS`, and `GAIN` at the top of [`realtime_audio.py`](realtime_audio.py).

To sanity-check the microphone alone (no Nengo / GUI):

```bash
python mic_input.py
```

## OS notes

The only OS-specific aspect is microphone permission:

- **macOS** will prompt to allow microphone access for your terminal app the first time you run it (System Settings → Privacy & Security → Microphone).
- **Windows**: ensure Settings → Privacy → Microphone allows desktop apps.
- **Linux**: typically just works once the user is in the `audio` group; otherwise grant ALSA / PulseAudio access through your distro's normal mechanisms.

## License

GPLv2 (see `LICENSE`). Matches Nengo's license.
