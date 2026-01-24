import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="TruthAnchor", page_icon="🛡️")

st.title("🛡️ TruthAnchor: News Integrity Verifier")
st.markdown("Enter a news headline or article snippet below to check its reliability.")

user_input = st.text_area("News Content:", height=150, placeholder="Paste text here...")

if st.button("Analyze Reliability"):
    if user_input:
        with st.spinner("Analyzing text..."):
            url = "http://127.0.0.1:8000/invocations"
            
            payload = {
                "instances": [{"text": user_input}]
            }
            
            try:
                response = requests.post(url, json=payload, timeout=10)
                
                if response.status_code == 200:
                    prediction = response.json()["predictions"][0]
                    label = prediction["label"]
                    score = prediction["score"] * 100

                    if label == "LABEL_0":
                        st.error(f" **UNRELIABLE / FAKE** ({score:.2f}% confidence)")
                    else:
                        st.success(f" **RELIABLE / REAL** ({score:.2f}% confidence)")
                else:
                    st.error(f"Server Error: {response.status_code}")
                    st.json(response.json()) 
                    
            except Exception as e:
                st.error(f"Connection Failed. Ensure the server is at {url}")
    else:
        st.warning("Please enter some text first.")