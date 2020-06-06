import streamlit as st
# from src.utils import utils
# from src.utils.reader import SciciteReader
# import configparser
# import os
# import spacy
# from spacy import displacy

# config = configparser.ConfigParser()
# config.read(os.path.join(os.path.dirname(__file__), os.pardir, "configs", "default.conf")

MODEL_NAMES = ["Model-1", "Model-2"]
DEFAULT_TEXT = "The regions of dhfr and dhps genes containing the mutations for antifolate resistance were amplified as described elsewhere (Plowe et al. 1995; Wang et al. 1997) and sequenced for detection of the mutations."
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

@st.cache(allow_output_mutation=True)
def load_model(name):
    # return spacy.load(name)
    pass

@st.cache(allow_output_mutation=True)
def process_text(text):
    # return nlp(text)
    pass

st.title("Interactive Citation Intent Classifier")
st.markdown(
    """
A little Streamlit app that lets you process text with different models and visualize the output and dependencies.
"""
)

model = st.selectbox("Model name", MODEL_NAMES)
# nlp = load_model(model)
# model_load_state.success(f"✅ Loaded model '{model}'.")
model_load_state = st.info(f"✅ Loaded model '{model}'.")

text = st.text_area("Text to analyze", DEFAULT_TEXT)
button_classify = st.button("Predict citation intent")
button_analyze = st.button("Analyze prediction")

doc = process_text(text)

if button_classify:
    # html = displacy.render(doc, style="ent")
    # st.write(HTML_WRAPPER.format(html))
    pass
if button_analyze:
    # html = displacy.render(doc)
    # # Double newlines seem to mess with the rendering
    # html = html.replace("\n\n", "\n")
    # st.write(HTML_WRAPPER.format(html))
    pass
