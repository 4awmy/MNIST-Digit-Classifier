"""MNIST drawing application. Start with: python -m streamlit run app.py"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')  # App inference matches CPU cloud hosting.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from canvas_component import digit_canvas
from model_io import MODEL_FILES, load_artifact, predict_digit
from preprocessing import DrawingError, decode_canvas, prepare_digit

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='MNIST · Draw a digit', page_icon='✍️', layout='centered')
st.markdown('''<style>
.stApp{background:#f7f8fc;color:#24283b}
.intro,.eyebrow,.prediction-card,.footer-note{font-family:Arial,sans-serif}
.block-container{max-width:920px;padding-top:3rem}
h1{font-size:3.1rem!important;letter-spacing:-1.8px;font-weight:750!important}
.eyebrow{color:#7566d8;font-size:11px;font-weight:700;letter-spacing:2.2px;margin-bottom:12px}
.intro{color:#787e90;font-size:16px;line-height:1.6;margin-bottom:26px}
.prediction-card{background:white;border:1px solid #e2e6ef;border-radius:22px;padding:28px;min-height:250px;text-align:center}
.prediction-label{font-size:11px;letter-spacing:1.5px;color:#8a8fa0;font-weight:700}
.prediction-digit{font-size:108px;line-height:1.25;font-weight:750;color:#6256dc;margin:10px 0}
.empty-digit{font-size:85px;color:#d8dbe7;line-height:1.8}
.prediction-caption{font-size:13px;color:#8a8fa0;line-height:1.6}
.footer-note{color:#9297a6;font-size:12px;margin-top:22px;border-top:1px solid #e7e9f1;padding-top:18px}
</style>''', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">MNIST / HANDWRITTEN DIGIT RECOGNITION</div>', unsafe_allow_html=True)
st.title('Draw a digit.')
st.markdown('<div class="intro">One sketch. Three ways to recognize it.<br>Choose a model, draw a single number from 0 to 9, and see its prediction.</div>', unsafe_allow_html=True)


def invalidate_prediction():
    st.session_state.pop('prediction', None)
    st.session_state.pop('processed_digit', None)
    st.session_state.pop('drawing_error', None)


@st.cache_resource(show_spinner=False)
def cached_model(name, path, modified_ns):
    return load_artifact(name, path)


models = {}
load_errors = []
for name, filename in MODEL_FILES.items():
    path = ROOT / 'models' / filename
    if path.is_file():
        try:
            models[name] = cached_model(name, str(path), path.stat().st_mtime_ns)
        except Exception as exc:
            load_errors.append(f'{name}: {type(exc).__name__}: {exc}')
if len(models) < 3:
    missing = [name for name in MODEL_FILES if name not in models]
    st.warning('Models still needed: ' + ', '.join(missing) + '. Add the missing saved model files to enable all three choices.')
if load_errors:
    st.error('A saved model could not be loaded. Check the model files and matching package versions.')
    with st.expander('Loading details'):
        for message in load_errors:
            st.text(message)

selection = st.selectbox('Recognition model', ['CNN', 'MLP', 'Random Forest'], key='model_choice',
                         on_change=invalidate_prediction)
left, right = st.columns([1.05, 1], gap='large')
with left:
    canvas_state = st.session_state.get('digit_canvas') or {}
    canvas = digit_canvas(
        key='digit_canvas', data={'image': canvas_state.get('drawing'), 'can_predict': selection in models},
        default={'drawing': None}, on_drawing_change=invalidate_prediction,
        on_clear_change=invalidate_prediction, on_predict_change=lambda: None,
        width='stretch', height='content',
    )
    if canvas.predict:
        invalidate_prediction()
        try:
            digit = prepare_digit(decode_canvas(canvas.predict['png']))
            st.session_state['prediction'] = predict_digit(models[selection], selection, digit)
            st.session_state['processed_digit'] = digit
        except DrawingError as exc:
            st.session_state['drawing_error'] = str(exc)
        except Exception:
            st.session_state['drawing_error'] = 'Prediction failed. Check the selected model and try again.'
    if st.session_state.get('drawing_error'):
        st.info(st.session_state['drawing_error'])

with right:
    result = st.session_state.get('prediction')
    if result:
        st.markdown(f'''<div class="prediction-card"><div class="prediction-label">PREDICTED DIGIT</div>
        <div class="prediction-digit">{result['digit']}</div>
        <div class="prediction-caption">{result['model']} · {result['milliseconds']:.1f} ms</div></div>''', unsafe_allow_html=True)
        with st.expander('What the model sees', expanded=True):
            preview, caption = st.columns([1, 2])
            preview.image(st.session_state['processed_digit'], width=84, clamp=True)
            caption.caption('Your drawing is cropped, centered, and resized to 28×28 pixels before prediction.')
        with st.expander('Scores for all digits'):
            st.bar_chart(pd.DataFrame({'Model score': result['scores']}, index=[str(i) for i in range(10)]))
            st.caption('These are model scores, not guaranteed confidence estimates.')
    else:
        st.markdown('''<div class="prediction-card"><div class="prediction-label">PREDICTION</div>
        <div class="empty-digit">?</div><div class="prediction-caption">Your result will appear here.<br>Draw a digit, then select Predict.</div></div>''', unsafe_allow_html=True)

with st.expander('Compare the three models'):
    path = ROOT / 'results' / 'comparison.csv'
    if path.is_file():
        table = pd.read_csv(path)
        st.dataframe(table[['model', 'test_accuracy', 'macro_f1', 'file_size_mib']], hide_index=True,
                     column_config={'test_accuracy': st.column_config.NumberColumn('Test accuracy', format='percent'),
                                    'macro_f1': st.column_config.NumberColumn('Macro F1', format='%.4f'),
                                    'file_size_mib': st.column_config.NumberColumn('Size (MiB)', format='%.2f')})
    else:
        st.caption('Model comparison data is unavailable.')
    st.caption('CNN uses image features, MLP uses fully connected layers, and Random Forest combines decision trees. MNIST test accuracy may differ from performance on your drawings.')
st.markdown('<div class="footer-note">Assignment 1 · Professional Training in AI II · Draw one digit at a time.</div>', unsafe_allow_html=True)
