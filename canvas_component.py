"""A small Streamlit v2 canvas with no third-party drawing dependency."""
from pathlib import Path
import streamlit as st

ASSETS = Path(__file__).resolve().parent / 'assets'
digit_canvas = st.components.v2.component(
    'mnist_digit_canvas',
    html=(ASSETS / 'canvas.html').read_text(encoding='utf-8'),
    css=(ASSETS / 'canvas.css').read_text(encoding='utf-8'),
    js=(ASSETS / 'canvas.js').read_text(encoding='utf-8'),
)
