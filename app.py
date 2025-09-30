import streamlit as st
import requests
#streamlit run app.py  
#uvicorn main:app --reload


#command about git on vscode
#.gitignore
#notepad .gitignore
#*.keras

st.title("Food Recognition App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    files = {"file": uploaded_file.getvalue()}
    response = requests.post("http://127.0.0.1:8000/predict", files=files)

    if response.status_code == 200:
        data = response.json()
        st.image(uploaded_file, caption=data["food_name"])
        st.write("Nutrition info:", data["nutrition"])
