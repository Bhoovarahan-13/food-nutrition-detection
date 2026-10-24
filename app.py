import streamlit as st, tempfile
from tracker import analyse
st.title("Smart Calorie Tracker")
f = st.file_uploader("Upload meal photo", type=["jpg", "jpeg", "png"])
hint = st.text_input("Optional description (e.g. 2 chapatis)")
if f and st.button("Analyse meal"):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as t:
        t.write(f.getbuffer())
    cap, items, df, total = analyse(t.name, text_hint=hint)
    st.image(t.name); st.write("**BLIP description:**", cap)
    st.write("**Detected items:**", ", ".join(items))
    st.table(df); st.success(f"Total: {total}")
