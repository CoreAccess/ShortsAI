import os
import tempfile
import streamlit as st
import torch
from Components.Edits import extractAudio, detect_face_and_crop
from Components.Helpers import get_file_hash
from Components.Transcriptions import transcribe_audio
from Components.SentimentAnalysis import analyze_emotions
from Components.Subtitles import write_srt, burn_subtitles
from Components.UserInterface import render_ui
from moviepy.video.io.VideoFileClip import VideoFileClip
import sys

# Render the UI and get user inputs
uploaded_file = render_ui()

# Create a directory named 'temp_files' if it doesn't exist
os.makedirs("temp_files", exist_ok=True)

# Check if a file is provided
if uploaded_file:

    # Initialize a variable for the temporary file path
    temp_file_path = None

    try:
        # *** Debugging Message *** #
        print("Video Processing Has Begun...")

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:

            # Write the uploaded file content to the temporary file
            temp_file.write(uploaded_file.read())

            # Get the path of the temporary file
            temp_file_path = temp_file.name

        # *** Debugging Message *** #
        print("Making Video Hash & Paths...")

        # Generate a hash for the video file
        file_hash = get_file_hash(temp_file_path)

        # Define the audio file path
        audio_path = f"temp_files/{file_hash}_audio.wav"

        # Define the transcript file path
        transcript_path = f"temp_files/{file_hash}_transcript.txt"

        # Define the emotion analysis file path
        emotion_path = f"temp_files/{file_hash}_emotions.txt"

        # *** Debugging Message *** #
        print("Starting The Audio Processing...")

        # Check if an audio file doesn't already exists
        if not os.path.exists(audio_path):
            # Load the video file
            with VideoFileClip(temp_file_path) as video:  # Load the video file
                audio_path = extractAudio(
                    temp_file_path, audio_path)  # Extract the audio

            # If the audio extraction failed
            if audio_path is None:
                # *** Debugging Message *** #
                print("Audio Extraction Failed. See Error Message Above.")

                # Exit the program
                sys.exit(1)

            # *** Debugging Message *** #
            print("Audio Processing Was a Success...")
        else:
            # *** Debugging Message *** #
            print("Audio File Already Exists; Skipping Extraction...")

        # *** Debugging Message *** #
        print("Starting Audio Transcription Process...")

        # Transcription using faster_whisper
        transcription_segments = transcribe_audio(
            audio_path, transcript_path, st, torch)  # Transcribe the audio

        # *** Debugging Message *** #
        print("Starting Sentiment Analysis Process...")

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

            # *** Debugging Message *** #
            print(
                f"Extracting Clip From {start_time:.2f}s To {end_time:.2f}s.")

            # Check if the cropped file exists
            if not os.path.exists(cropped_file):
                # Detect face and crop using clip range
                detect_face_and_crop(
                    temp_file_path, cropped_file, start_time, end_time)

                # *** Debugging Message *** #
                print(f"Clip Was Extracted To: {cropped_file}")
            else:
                # *** Debugging Message *** #
                print(f"Clip Already Exists Using: {cropped_file}")

            # Create subtitles
            subtitles = [
                (segment["timestamp"][0] - start_time, segment["timestamp"][1] - start_time,
                 segment["text"].replace('\u266a', '*'))
                for segment in transcription_segments
                if segment["timestamp"][0] >= start_time and segment["timestamp"][1] <= end_time
            ]

            # Define the SRT file path
            srt_file = f"temp_files/{file_hash}_subtitles.srt"

            # *** Debugging Message *** #
            print("Starting Subtitle Generation...")

            # Write the subtitles to the SRT file
            write_srt(subtitles, srt_file)

            # If the subtitled file doesn't exists
            if not os.path.exists(subtitled_file):
                # Burn the subtitles
                burn_subtitles(cropped_file, srt_file, subtitled_file)

                # *** Debugging Message *** #
                print(
                    f"Generated Clip With Subtitles At: {subtitled_file}")
            else:
                # *** Debugging Message *** #
                print(f"Existing Clip Already Exists Using: {subtitled_file}")

            # *** Debugging Message *** #
            print(f"Final Clip Ready For Viewing At: {subtitled_file}")
        else:
            # *** Debugging Message *** #
            print("No Dramatic Segments Detected...")

        # *** Debugging Message *** #
        print("Processing Completed Successfully! Go Watch The Clip!")

    except Exception as e:
        # *** Debugging Message *** #
        print(f"There Was a Processing Error: {e}")
    finally:
        # Check if the temporary file exists
        if temp_file_path and os.path.exists(temp_file_path):
            # Remove the temporary file
            os.remove(temp_file_path)

            # *** Debugging Message *** #
            print(f"Temporary File Has Been Deleted From: {temp_file_path}")
