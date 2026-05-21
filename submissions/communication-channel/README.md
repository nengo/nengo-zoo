# communication-channel

Two ensembles wired in series — value in, value out.

## Description
this is a test edit

The simplest useful Nengo network. A scalar input is encoded by ensemble `a`, decoded as the identity function, and passed to ensemble `b`, which represents the same value with its own population activity. Decoding `b` recovers the original signal, demonstrating that a value can move through neural populations without dedicated wiring.

This is the "hello world" of the Neural Engineering Framework.

## Run it

Open in NengoGUI:

```bash
pip install -r requirements.txt nengo-gui
nengo communication_channel.py
```

## License

GPLv2 (see `LICENSE`).
