import streamlit as st
import pandas as pd
import joblib
import sklearn 

st.set_page_config(layout="wide")

pipe_lr = joblib.load(open("emotion_classifier_pipe_lr.pkl", "rb"))

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

emotions_trad = {"anger":"Colére","disgust":"Dégout", "fear":"peur", "happy":"heureux", "joy":"Joie", "neutral":"Neutre", "sad":"Triste", "sadness":"Tristesse", "shame":"Honte", "surprise":"Surprise"}
a,b,c = st.columns(3)
with b:
    st.title("Analyse Emotion Tweet")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Extrait Tweet for Test")
    data = pd.read_csv("jupyter/source.csv")
    # Échantillonnage aléatoire de 5 tweets
    sample_tweets = data.sample(n=3)

    # Affichage des tweets échantillonnés
    for index, row in sample_tweets.iterrows():
        emotion = row["Emotion"]
        text = row["Text"]
        st.write(text)
        st.divider()


    st.subheader("Data Source")
    data1 = pd.read_csv("jupyter/source.csv", index_col=0)
    st.dataframe(data1)

with col2:
    st.subheader("About")
    st.write("TA linear regression model from the Scikit-learn library was used to train a dataset consisting of speeches and their respective emotions. Joblib was employed to store and utilize the trained model on the website.")
    st.caption('Cornélius Vincent @2023')
    
    txt = st.text_area(
        "Tweet to analyze",
    )
    btn = st.button("analyser")

    if btn and txt != "":
        # st.write("Sentiment:", predict_emotions(txt))
        ########
        r = get_prediction_proba(txt)
        proba_df = pd.DataFrame(r, columns=pipe_lr.classes_)
        proba_df = proba_df.transpose()
        proba_df_clean = proba_df.transpose()
        # st.write(proba_df_clean)
        melted_df = proba_df_clean.melt(var_name="Emotion", value_name="Proba")
        # st.write(melted_df)
        st.bar_chart(proba_df)

st.markdown(
    """
    <style>
    .centered-link {
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.write("<div class='centered-link'><a href='https://www.linkedin.com/in/corneliusvincent/' target='blank_'>Mon profil LinkedIn</div>", unsafe_allow_html=True)
