# Import the WhisperModel class from the faster_whisper library
from faster_whisper import WhisperModel


def transcribe_audio(audio_path, transcript_path, st, torch):
    # Define a function called transcribe_audio that takes the audio path, transcript path, streamlit object, and torch module as input
    """
    Transcribe the audio using faster_whisper and save the transcript.
    """
    if not __import__("os").path.exists(transcript_path):
        # Check if the transcript file does not exist
        with st.spinner("Transcribing audio..."):
            # Display a spinner in the streamlit app while transcribing audio
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            # Determine the device to use for transcription (CUDA if available, otherwise CPU)
            # Load the Whisper model
            model = WhisperModel("base.en", device=device_str)
            # Load the Whisper model with the specified model size and device
            segments, _ = model.transcribe(
                audio_path,
                beam_size=5,
                language="en",
                max_new_tokens=128,
                condition_on_previous_text=False
            )
            # Transcribe the audio using the Whisper model with the specified parameters
            segments = list(segments)
            # Convert the segments iterator to a list
            transcription_segments = []
            # Initialize an empty list to store the transcription segments
            transcription = ""
            # Initialize an empty string to store the complete transcription
            for seg in segments:
                # Iterate over each segment in the transcription
                start, end, text = seg.start, seg.end, seg.text.replace(
                    '\u266a', '*')
                # Extract the start time, end time, and text from the segment, replacing musical notes with asterisks
                transcription_segments.append(
                    {"timestamp": [start, end], "text": text})
                # Append the segment information to the list of transcription segments
                transcription += f"[{start:.2f} - {end:.2f}] {text}\n"
                # Append the segment text to the complete transcription string with timestamps
            with open(transcript_path, 'w', encoding='utf-8') as f:
                # Open the transcript file in write mode with UTF-8 encoding
                f.write(transcription)
                # Write the complete transcription to the file
        st.write("Transcription complete!")
        # Write a message to the streamlit app indicating that the transcription is complete
    else:
        # If the transcript file exists
        from Components.Helpers import load_transcription_segments
        # Import the load_transcription_segments function
        transcription_segments = load_transcription_segments(transcript_path)
        # Load the transcription segments from the file
        st.write("Using existing transcript.")
        # Write a message to the streamlit app indicating that an existing transcript is being used
    return transcription_segments
# Return the transcription segments
