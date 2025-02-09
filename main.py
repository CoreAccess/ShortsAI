import os  # Import the os module for interacting with the operating system
import tempfile  # Import the tempfile module for creating temporary files
import streamlit as st  # Import the Streamlit library
import requests  # Import the requests library
import torch  # Import the torch library
# Import components
# Import functions for audio extraction and video cropping
from Components.Edits import extractAudio, crop_video, detect_face_and_crop
from Components.Helpers import get_file_hash  # Import the get_file_hash function
# Import the transcribe_audio function
from Components.Transcriptions import transcribe_audio
# Import the analyze_emotions function
from Components.SentimentAnalysis import analyze_emotions
# Import functions for subtitle creation and burning
from Components.Subtitles import write_srt, burn_subtitles
from Components.UserInterface import render_ui  # Import the render_ui function

# UI input moved to Components/UserInterface.py
user_input, uploaded_file = render_ui()  # Render the UI and get user inputs

# Create a directory named 'saved_files' if it doesn't exist
os.makedirs("saved_files", exist_ok=True)

if user_input and uploaded_file:  # Check if both a URL and a file are provided
    # Display an error message
    st.error("Provide a YouTube URL or upload a file, not both.")
elif user_input or uploaded_file:  # Check if either a URL or a file is provided
    temp_file_path = None  # Initialize a variable for the temporary file path
    try:
        st.write("Starting audio extraction...")  # Display a message
        if user_input:  # If a YouTube URL is provided
            video_url = user_input  # Get the URL from the input field
            video_response = requests.get(video_url)  # Download the video
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                # Write the video content to the temporary file
                temp_file.write(video_response.content)
                temp_file_path = temp_file.name  # Get the path of the temporary file
            # Display a message
            st.write(f"Downloaded video to temporary file: {temp_file_path}")
        else:  # If a local file is provided
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                # Write the uploaded file content to the temporary file
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name  # Get the path of the temporary file
            st.write(
                # Display a message
                f"Saved uploaded file to temporary file: {temp_file_path}")

        # Generate a hash for the video file
        file_hash = get_file_hash(temp_file_path)
        # Define the audio file path
        audio_path = f"saved_files/{file_hash}_audio.wav"
        # Define the transcript file path
        transcript_path = f"saved_files/{file_hash}_transcript.txt"
        # Define the emotion analysis file path
        emotion_path = f"saved_files/{file_hash}_emotions.txt"

        # Audio extraction
        # Check if the audio file already exists
        if not os.path.exists(audio_path):
            # Import the VideoFileClip class
            from moviepy.video.io.VideoFileClip import VideoFileClip
            with VideoFileClip(temp_file_path) as video:  # Load the video file
                audio_path = extractAudio(
                    temp_file_path, audio_path)  # Extract the audio
            st.write("Audio extraction complete!")  # Display a message
        else:
            # Display a message
            st.write("Audio already exists; skipping extraction.")

        # Transcription using faster_whisper
        transcription_segments = transcribe_audio(
            audio_path, transcript_path, st, torch)  # Transcribe the audio

        # Emotion analysis using transformers pipeline
        device_id = 0 if torch.cuda.is_available() else -1  # Determine the device to use
        emotions = analyze_emotions(
            transcription_segments, emotion_path, st, torch, device_id)  # Analyze the emotions

        dramatic_segments = [segment for segment in emotions if segment['label'] in [
            'anger', 'fear', 'sadness']]  # Filter for dramatic segments
        if dramatic_segments:  # If dramatic segments are found
            # Get the start time
            start_time = float(dramatic_segments[0]['start'])
            end_time = start_time + 59.0  # Calculate the end time
            st.write(
                # Display a message
                f"Extracting clip from {start_time:.2f}s to {end_time:.2f}s.")

            for segment in transcription_segments:  # Iterate over the transcription segments
                if segment["timestamp"][1] >= start_time:  # Find the starting sentence
                    # Update the start time
                    start_time = segment["timestamp"][0]
                    break

            # Define the cropped file path
            cropped_file = f"saved_files/{file_hash}_dramatic_clip.mp4"
            # Define the subtitled file path
            subtitled_file = f"saved_files/{file_hash}_dramatic_clip_with_subtitles.mp4"

            # Check if the cropped file exists
            if not os.path.exists(cropped_file):
                detect_face_and_crop(
                    # Detect face and crop using clip range
                    temp_file_path, cropped_file, start_time, end_time)
                st.write(f"Extracted clip: {cropped_file}")
            else:
                # Display a message
                st.write(f"Reusing cropped clip: {cropped_file}")

            subtitles = [
                (segment["timestamp"][0] - start_time, segment["timestamp"][1] - start_time,
                 segment["text"].replace('\u266a', '*'))
                for segment in transcription_segments
                if segment["timestamp"][0] >= start_time and segment["timestamp"][1] <= end_time
            ]  # Create subtitles
            # Define the SRT file path
            srt_file = f"saved_files/{file_hash}_subtitles.srt"
            # Write the subtitles to the SRT file
            write_srt(subtitles, srt_file)
            # Check if the subtitled file exists
            if not os.path.exists(subtitled_file):
                # Burn the subtitles
                burn_subtitles(cropped_file, srt_file, subtitled_file)
                st.write(
                    # Display a message
                    f"Generated clip with burnt subtitles: {subtitled_file}")
            else:
                # Display a message
                st.write(f"Using existing subtitled clip: {subtitled_file}")

            # Display a success message
            st.success(f"Final output ready: {subtitled_file}")
        else:
            st.write("No dramatic segments detected.")  # Display a message

        # Display a success message
        st.success("Processing completed successfully!")

    except Exception as e:
        # Display a processing error message
        st.error(f"Processing error: {e}")
    finally:
        # Check if the temporary file exists
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)  # Remove the temporary file
            # Display a message
            st.write(f"Deleted temporary file {temp_file_path}")
