import streamlit as st
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


@st.cache_resource
def translation_model():
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    return tokenizer, model


tokenizer, model = translation_model()
st.title("Wanna know French translation to your English ? Here's your way to go !")
st.write("Enter the text in English to translate in French")

text_input = st.text_input("Input Text", "Hello, how are you ?")

if st.button("Translate"):
    inputs = tokenizer.encode(text_input, return_tensors="pt")
    outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write(f"Translated Text (French): {translated_text}")
