import streamlit as st
import pickle
import requests

#######################################################

# The code below is to control the layout width of the app.
if "widen" not in st.session_state:
    layout = "centered"
else:
    layout = "wide" if st.session_state.widen else "centered"

#######################################################

# The code below is for the title and logo.
title = "Offensive Tweet Classifier"
st.set_page_config(layout=layout, page_title=title, page_icon="🤗")
# The block of code below is to display the title, logos and introduce the app.

c1, c2 = st.columns([0.4, 2])

with c1:

    st.image(
        "logo.png",
        width=110,
    )

with c2:

    st.caption("")
    st.title(title)
#######################################################

st.sidebar.header('About')
st.markdown("""
Classify Tweet on-the-fly with this mighty app. Check if your tweet is `Offensive` or `Not Offensive`. 🚀
""")
st.sidebar.markdown("""
App is created using 🎈[Streamlit](https://streamlit.io/) and [HuggingFace](https://huggingface.co/inference-api)'s [TimeLMs offensive tweet](https://huggingface.co/rifatmonzur/offensiveTweet) model.
""")
st.sidebar.markdown("""
[OLID dataset](https://www.kaggle.com/datasets/feyzazkefe/olid-dataset/data) is used to finetune [TimeLMs](https://huggingface.co/cardiffnlp/twitter-roberta-base-offensive)
""")
st.sidebar.markdown("""
Developed by [Rifat Monzur](https://www.linkedin.com/in/rifatmonzur/)
""")

st.sidebar.header("Resources")
st.sidebar.markdown(
    """
- [Source Code](https://github.com/rifat1234/offensive-tweet)
- [Project Report](https://www.overleaf.com/read/jbszjmptxtzd#9ea583)
- [Hugging Face Model Inference API](https://huggingface.co/rifatmonzur/offensiveTweet)
"""
)

with st.form("my_form"):
    model_option_1 = 'TimeLMs (Huggingface LLM Transformer)'
    model_option_2 = 'Multilayer perceptron'
    offensive_tweet = "@USER is so unattractive. I understand why her husband left her."
    option = st.selectbox(
        'Choose your preferred model',
        (model_option_1, model_option_2))
    txt = st.text_area("Write your own tweet and check if it is offensive or not",
                       value=offensive_tweet, max_chars=140)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")

    def print_verdict_message(label):
        if label == 'OFF':
            st.success("This tweet is :red[**Offensive**]")
        elif label == 'NOT':
            st.success("This tweet is :green[**not Offensive**]")
        else:
            st.error("Check your internet connection")
    if submitted:
        with st.spinner('Please wait...'):
            if len(txt.strip()) == 0:
                st.error("Tweet need to have at least one character")
            elif option == model_option_2:
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

                API_TOKEN = st.secrets['API_TOKEN']
                headers = {"Authorization": f"Bearer {API_TOKEN}"}
                API_URL = "https://api-inference.huggingface.co/models/rifatmonzur/offensiveTweet"

                try:
                    data = query({
                        "inputs": f"{preprocess(txt)}",
                        "options": {"wait_for_model": True}
                    })
                    verdict = 'OFF' if data[0][0]['label'] == 'offensive' else 'NOT'
                    print_verdict_message(verdict)
                except:
                    print_verdict_message("error")

    else:
        if txt == offensive_tweet:
            print_verdict_message('OFF')

