"""Cross-platform microphone capture for use as a Nengo node input.

Uses the sounddevice library (PortAudio), whose pip wheels bundle the
native binaries on macOS and Windows, so no OS-specific setup is needed.

Audio is written by the PortAudio callback thread into a ring buffer;
`get_window` reads the most recent samples, so the model is always "live"
regardless of how fast or slow the simulation is running.
"""

import threading

import numpy as np
import sounddevice as sd


class MicrophoneInput:
    """Captures mono microphone audio into a rolling buffer.

    Parameters
    ----------
    buffer_seconds : float
        Length of audio history to keep.
    gain : float
        Multiplier applied to samples in `get_window`. Raw mic samples are
        typically small (|x| < 0.1), so gain brings them into the +/-1
        radius of a default ensemble.
    """

    def __init__(self, buffer_seconds=1.0, gain=10.0):
        self.gain = gain

        try:
            device_info = sd.query_devices(kind="input")
        except (sd.PortAudioError, ValueError) as err:
            raise RuntimeError(
                "No microphone / audio input device found. "
                "Check that a mic is connected and that this application "
                "has permission to use it."
            ) from err

        # Use the device's default rate; forcing a rate the device does
        # not support is the main cross-platform failure mode.
        self.samplerate = int(device_info["default_samplerate"])

        self._lock = threading.Lock()
        self._buffer = np.zeros(int(buffer_seconds * self.samplerate))
        self._write_idx = 0

        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, time_info, status):
        samples = indata[:, 0]
        n = len(self._buffer)
        with self._lock:
            idx = self._write_idx
            if frames >= n:
                self._buffer[:] = samples[-n:]
                self._write_idx = 0
            else:
                end = idx + frames
                if end <= n:
                    self._buffer[idx:end] = samples
                else:
                    split = n - idx
                    self._buffer[idx:] = samples[:split]
                    self._buffer[: end - n] = samples[split:]
                self._write_idx = end % n

    def get_window(self, dims, window_ms):
        """Return the most recent `window_ms` of audio as `dims` values.

        The raw window is resampled to exactly `dims` points so the
        output dimensionality is independent of the device sample rate.
        """
        n_samples = max(2, int(self.samplerate * window_ms / 1000.0))
        n = len(self._buffer)
        with self._lock:
            idx = self._write_idx
            window = np.roll(self._buffer, -idx)[-n_samples:]
        resampled = np.interp(
            np.linspace(0.0, 1.0, dims),
            np.linspace(0.0, 1.0, n_samples),
            window,
        )
        return self.gain * resampled

    def make_node_fn(self, dims, window_ms):
        """Return an `f(t)` suitable for `nengo.Node(f, size_out=dims)`."""

        def node_fn(t):
            return self.get_window(dims, window_ms)

        return node_fn

    def close(self):
        self._stream.stop()
        self._stream.close()


_mic = None


def get_mic(**kwargs):
    """Singleton accessor for the microphone.

    nengo_gui re-executes the model script on every edit/reload, but this
    module stays cached in sys.modules, so reloads reuse the open stream
    instead of stacking up new ones.
    """
    global _mic
    if _mic is None:
        _mic = MicrophoneInput(**kwargs)
    return _mic


if __name__ == "__main__":
    import time

    mic = get_mic()
    print(f"Capturing at {mic.samplerate} Hz. Make some noise...")
    for _ in range(10):
        time.sleep(0.2)
        rms = np.sqrt(np.mean(mic.get_window(64, 100) ** 2))
        print(f"RMS (gained): {rms:.4f}")
    mic.close()
