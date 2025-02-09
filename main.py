import os  # Import the os module for interacting with the operating system
import tempfile  # Import the tempfile module for creating temporary files
import streamlit as st  # Import the Streamlit library
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
from moviepy.video.io.VideoFileClip import VideoFileClip

# Render the UI and get user inputs
uploaded_file = render_ui()

# Create a directory named 'temp_files' if it doesn't exist
os.makedirs("temp_files", exist_ok=True)

# Check if a file is provided
if uploaded_file:
    temp_file_path = None  # Initialize a variable for the temporary file path
    try:
        st.write("Starting audio extraction...")  # Display a message

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            # Write the uploaded file content to the temporary file
            temp_file.write(uploaded_file.read())

            # Get the path of the temporary file
            temp_file_path = temp_file.name

        # Display a message
        st.write(
            f"Saved uploaded file to temporary file: {temp_file_path}")

        # Generate a hash for the video file
        file_hash = get_file_hash(temp_file_path)

        # Define the audio file path
        audio_path = f"temp_files/{file_hash}_audio.wav"

        # Define the transcript file path
        transcript_path = f"temp_files/{file_hash}_transcript.txt"

        # Define the emotion analysis file path
        emotion_path = f"temp_files/{file_hash}_emotions.txt"

        # Check if an audio file doesn't already exists
        if not os.path.exists(audio_path):
            # Load the video file
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
        emotions = analyze_emotions(
            transcription_segments, emotion_path, st, torch)  # Analyze the emotions

        # Filter for dramatic segments
        dramatic_segments = [segment for segment in emotions if segment['label'] in [
            'anger', 'fear', 'sadness']]

        # If dramatic segments are found
        if dramatic_segments:
            # Get the start time
            start_time = float(dramatic_segments[0]['start'])

            # Calculate the end time
            end_time = start_time + 59.0

            # Display a message
            st.write(
                f"Extracting clip from {start_time:.2f}s to {end_time:.2f}s.")

            # Iterate over the transcription segments
            for segment in transcription_segments:
                # Update the start time
                if segment["timestamp"][1] >= start_time:
                    start_time = segment["timestamp"][0]
                    break

            # Define the cropped file path
            cropped_file = f"temp_files/{file_hash}_dramatic_clip.mp4"

            # Define the subtitled file path
            subtitled_file = f"temp_files/{file_hash}_dramatic_clip_with_subtitles.mp4"

            # Check if the cropped file exists
            if not os.path.exists(cropped_file):
                # Detect face and crop using clip range
                detect_face_and_crop(
                    temp_file_path, cropped_file, start_time, end_time)

                # Display a message
                st.write(f"Extracted clip: {cropped_file}")
            else:
                # Display a message
                st.write(f"Reusing cropped clip: {cropped_file}")

            # Create subtitles
            subtitles = [
                (segment["timestamp"][0] - start_time, segment["timestamp"][1] - start_time,
                 segment["text"].replace('\u266a', '*'))
                for segment in transcription_segments
                if segment["timestamp"][0] >= start_time and segment["timestamp"][1] <= end_time
            ]

            # Define the SRT file path
            srt_file = f"temp_files/{file_hash}_subtitles.srt"

            # Write the subtitles to the SRT file
            write_srt(subtitles, srt_file)

            # If the subtitled file doesn't exists
            if not os.path.exists(subtitled_file):
                # Burn the subtitles
                burn_subtitles(cropped_file, srt_file, subtitled_file)

                # Display a message
                st.write(
                    f"Generated clip with burnt subtitles: {subtitled_file}")
            else:
                # Display a message
                st.write(f"Using existing subtitled clip: {subtitled_file}")

            # Display a success message
            st.success(f"Final output ready: {subtitled_file}")
        else:
            # Display a message
            st.write("No dramatic segments detected.")

        # Display a success message
        st.success("Processing completed successfully!")

    except Exception as e:
        # Display a processing error message
        st.error(f"Processing error: {e}")
    finally:
        # Check if the temporary file exists
        if temp_file_path and os.path.exists(temp_file_path):
            # Remove the temporary file
            os.remove(temp_file_path)

            # Display a message
            st.write(f"Deleted temporary file {temp_file_path}")
