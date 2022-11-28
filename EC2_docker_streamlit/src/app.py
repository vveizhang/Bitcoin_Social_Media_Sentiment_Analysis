import streamlit as st
from funcs import load_model, input_fn, run_model

st.title("Bert-based bitcoin social media sentiment analysis")
st.write('This app uses Bert to identify the sentiment of any bitcoin social media')
st.markdown('The source code for this app can be found in this GitHub repo: [GPT2-News-Classifier](https://github.com/vveizhang/transformer_predict_circRNA).')

example_text = """
"I think bitcoin is worthless. I will never buy it"
"""

input_text = st.text_area(
    label="Input/Paste News here:",
    value="",
    height=30,
    placeholder="Example:{}".format(example_text)
    )

# load model here to save
model = load_model()

if input_text == "":
    input_text = example_text

if st.button("Run Bert!"):
    if len(input_text) < 300:
        st.write("Please input more text!")
    else:
        with st.spinner("Running..."):

            model_input = input_fn(input_text)
            model_output = run_model(model, *model_input)
            st.write("Predicted sentiment:")
            st.write(model_output)