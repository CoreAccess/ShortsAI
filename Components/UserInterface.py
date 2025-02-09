import streamlit as st


def render_ui():
    """
    Render UI styling and return the YouTube URL and uploaded file.
    """
    st.title("Simple Video Short Maker")
    user_input = st.text_input("Enter a YouTube URL")
    uploaded_file = st.file_uploader("Choose a video file (Local)",
                                     type=['mkv', 'mp4', 'mov', 'avi'])
    return user_input, uploaded_file
