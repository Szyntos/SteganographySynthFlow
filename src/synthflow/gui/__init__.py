"""SSF rack GUI package.

Run with:  python -m gui

Layout of the package (strict UI ↔ DSP separation):
  engines.py       audio streaming + note gating around EncoderDSP/DecoderDSP
                   (no tkinter)
  theme.py         colours + ttk styles
  widgets.py       reusable UI building blocks (no DSP)
  encoder_panel.py / decoder_panel.py / keyboard_bar.py   rack modules
  app.py           the rack window wiring modules to an engine
"""
