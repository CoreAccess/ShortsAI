from faster_whisper import WhisperModel
from Components.Helpers import load_transcription_segments

# ----------------------------------------------------------------
# Transcribe audio using faster_whisper and save the transcript.
# ----------------------------------------------------------------


def transcribe_audio(audio_path, transcript_path, st, torch):
    # If no transcript file exists, transcribe the audio
    if not __import__("os").path.exists(transcript_path):
        with st.spinner("Transcribing audio..."):  # Display a spinner

            device_str = "cuda" if torch.cuda.is_available() else "cpu"

            # Load the Whisper model
            model = WhisperModel("base.en", device=device_str)

            # Transcribe the audio using the Whisper model
            segments, _ = model.transcribe(
                audio_path,
                beam_size=5,
                language="en",
                max_new_tokens=128,
                condition_on_previous_text=False
            )

            # Convert the segments iterator to a list
            segments = list(segments)

            # Initialize an empty list to store the transcription segments
            transcription_segments = []

            # Initialize an empty string to store the complete transcription
            transcription = ""

            # Iterate over each segment in the transcription
            for seg in segments:
                # Replace musical notes with empty strings
                start, end, text = seg.start, seg.end, seg.text.replace(
                    '\u266a', '')

                # Append the segment information to the list of transcription segments
                transcription_segments.append(
                    {"timestamp": [start, end], "text": text})

                # Append the segment text to the complete transcription string with timestamps
                transcription += f"[{start:.2f} - {end:.2f}] {text}\n"

            # Open the transcript file in write mode with UTF-8 encoding
            with open(transcript_path, 'w', encoding='utf-8') as f:

                # Write the complete transcription to the file
                f.write(transcription)

        st.write("Transcription complete!")  # Display a message
    else:  # If the transcript file exists

        # Load the transcription segments from the file
        transcription_segments = load_transcription_segments(transcript_path)

        st.write("Using existing transcript.")  # Display a message
    return transcription_segments
