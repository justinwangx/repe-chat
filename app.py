import json
import requests
import streamlit as st

use_openai = False
if use_openai:
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

EMOTION_TO_EMOJI = {
    "happiness": ["😢", "😄"],
    "anger": ["😌", "😡"],
    "surprise": ["😐", "😲"],
    "fear": ["🐱", "🙀"],
    "disgust": ["😻", "🤢"]
}

ROLE_TO_EMOJI = {
    "user": "😛",
    # ideally we'd have a robot emoji here but looks like the `avatar` kwarg isn't supported in st.write_stream
    "assistant": None
}

def generate_stream(response):
    for chunk in response.iter_lines():
        if chunk:
            try:
                yield json.loads(chunk.decode().replace("data: ", ""))["choices"][0]["delta"]["content"]
            except Exception:
                continue

st.set_page_config(page_title="RepE Chat", page_icon="favicon.ico")
st.title("RepE Chat 🤯")

with st.sidebar:
    st.header("About")
    st.write("""
        Chat with a rep-controlled model! 

        You can now stimulate regions in Mistral-7B-Instruct-v0.2's brain while talking to it. 
        
        Using [Representation Engineering](%s), we found directions within the model's activation space that correspond to particular emotions. 
        
        Without any prompt engineering, we can use these directions at inference-time to control the model's responses!
    """ % "https://www.ai-transparency.org/")

emotion = st.selectbox(
    "Emotion",
    ("Happiness", "Anger", "Surprise", "Fear", "Disgust")
)

col1, col2, col3 = st.columns([1, 8, 1])

with col1:
    st.markdown(f'<div style="font-size: 30px;">{EMOTION_TO_EMOJI[emotion.lower()][0]}</div>', unsafe_allow_html=True)

with col2:
    repe_coefficient = st.slider("RepE coefficient", -1.5, 1.5, value=0.0)

with col3:
    st.markdown(f'<div style="font-size: 30px; text-align: right;">{EMOTION_TO_EMOJI[emotion.lower()][1]}</div>', unsafe_allow_html=True)

st.write('<hr style="border: 2px solid #e0d8d7;"></hr>', unsafe_allow_html=True)

# Chat Logic

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=ROLE_TO_EMOJI[message["role"]]):
        st.markdown(message["content"])

if prompt := st.chat_input("Say something"):
    with st.chat_message("user", avatar=ROLE_TO_EMOJI["user"]):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        if use_openai:
            if "openai_model" not in st.session_state:
                st.session_state["openai_model"] = "gpt-3.5-turbo"

            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True
            )

            response = st.write_stream(stream)
        else:
            r = requests.post(
                url=st.secrets["MODEL_ENDPOINT"],
                headers={"Content-Type": "application/json"},
                json={
                    "model": "rep-control",
                    "messages": st.session_state.messages,
                    "n": 1,
                    "echo": False,
                    "logprobs": False,
                    "stream": True,
                    "control": emotion.lower(),
                    "repe_coefficient": repe_coefficient
                },
                stream=True
            )

            response = st.write_stream(generate_stream(r))

    st.session_state.messages.append({"role": "assistant", "content": response})
