# controlled-oscillator

A 2D oscillator whose angular speed is gated by a third recurrent dimension — and that gate, in turn, is driven by a separate input ensemble. A clean demonstration of one neural population controlling the dynamic regime of another.

## Description

This is the controlled-oscillator pattern from the CNRG tutorial series. The differential equations are the same as a plain oscillator — a 2D circle in state space — but the angular speed `s` is itself a dimension of the same ensemble:

```
dx0/dt = -x1 · s + x0 · (r - x0² - x1²)
dx1/dt =  x0 · s + x1 · (r - x0² - x1²)
```

A separate `speed` ensemble feeds the third dimension `x[2]`. Slide the input slider in NengoGUI and watch the oscillator speed up or stop.

## Run it

In NengoGUI:

```bash
pip install nengo-gui
nengo controlled_oscillator.py
```

## License

GPLv2 (see `LICENSE`).
