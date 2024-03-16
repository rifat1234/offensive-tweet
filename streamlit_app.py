import streamlit as st
import pickle

@st.cache_resource
def load_model(model_file, vectorizer_file):
    model = pickle.load(open(model_file, 'rb'))
    vectorizer = pickle.load(open(vectorizer_file, 'rb'))
    return model, vectorizer

def prepare_dataset(tweets, count_vectorizer):
    values = count_vectorizer.transform(tweets)
    return values

st.subheader("Offensive tweet detector")
with st.form("my_form"):
    txt = st.text_area("Write your own tweet and check if it is offensive or not", max_chars=140)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        model, vectorizer = load_model('model.sav', 'vectorizer.sav')
        values = prepare_dataset([txt], vectorizer)
        pred = model.predict(values)
        if pred == "OFF":
            st.markdown("This tweet is :red[**Offensive**]")
        else:
            st.markdown("This tweet is :green[**not Offensive**]")