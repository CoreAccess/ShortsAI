import streamlit as st

################################################################################
#                                                                              #
#             Render UI styling and return the YouTube URL                     #
#                       and uploaded file.                                     #
#                                                                              #
################################################################################


def render_ui():
    st.title("Simple Video Short Maker")  # Set the title of the app

    # Provide a video upload option to the user
    uploaded_file = st.file_uploader("Choose a video file (Local)",
                                     type=['mkv', 'mp4', 'mov', 'avi'])
    return uploaded_file
