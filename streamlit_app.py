import streamlit as st
import pickle
import requests

st.subheader("Offensive tweet detector")
with st.form("my_form"):
    model_option_1 = 'TimeLMs (Huggingface LLM Transformer)'
    model_option_2 = 'Multilayer perceptron'
    option = st.selectbox(
        'Choose your preferred model',
        (model_option_1, model_option_2))
    txt = st.text_area("Write your own tweet and check if it is offensive or not", max_chars=140)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")

    def print_verdict_message(label):
        if label == 'OFF':
            st.markdown("This tweet is :red[**Offensive**]")
        else:
            st.markdown("This tweet is :green[**not Offensive**]")

    if submitted:
        if option == model_option_2:
            @st.cache_resource
            def load_model(model_file, vectorizer_file):
                model = pickle.load(open(model_file, 'rb'))
                vectorizer = pickle.load(open(vectorizer_file, 'rb'))
                return model, vectorizer

            def prepare_dataset(tweets, count_vectorizer):
                values = count_vectorizer.transform(tweets)
                return values

            model, vectorizer = load_model('model1/model.sav', 'model1/vectorizer.sav')
            values = prepare_dataset([txt], vectorizer)
            pred = model.predict(values)
            print_verdict_message(pred)
        else:
            def preprocess(text):
                preprocessed_text = []
                for t in text.split():
                    if len(t) > 1:
                        t = '@user' if t[0] == '@' and t.count('@') == 1 else t
                        t = 'http' if t.startswith('http') else t
                    preprocessed_text.append(t)
                return ' '.join(preprocessed_text)

            def query(payload):
                response = requests.post(API_URL, headers=headers, json=payload)
                return response.json()

            API_TOKEN = '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'#st.secrets['API_TOKEN']
            headers = {"Authorization": f"Bearer {API_TOKEN}"}
            API_URL = "https://api-inference.huggingface.co/models/rifatmonzur/offensiveTweet"

            data = query({
                "inputs": f"{preprocess(txt)}",
                "options": {"wait_for_model": True}
            })

            verdict = 'OFF' if data[0][0]['label'] == 'offensive' else 'NOT'
            print_verdict_message(verdict)

