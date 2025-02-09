from transformers import pipeline
import os
from Components.Helpers import chunk_text_with_timestamps, save_emotion_analysis, load_emotion_analysis

# ----------------------------------------------------------------------------
# Analyze emotions from the transcription segments using an emotion classifier.
# ----------------------------------------------------------------------------


def analyze_emotions(transcription_segments, emotion_path, st, torch):
    # If an emotional analysis file does not exist
    if not os.path.exists(emotion_path):
        with st.spinner("Analyzing emotions..."):  # Display a spinner
            # Determine the device to use
            device_str = "cuda" if torch.cuda.is_available() else "cpu"

            # Create a pipeline for text classification
            emotion_analyzer = pipeline(
                "text-classification",
                model="michellejieli/emotion_text_classifier",
                device=device_str
            )

            # Chunk the transcription segments into smaller pieces with timestamps
            transcription_chunks = chunk_text_with_timestamps(
                transcription_segments)

            # Initialize an empty list to store the emotion analysis results
            emotions = []

            # Iterate over each chunk of transcription segments
            for chunk in transcription_chunks:
                # Extract the text from the chunk and join it into a single string
                chunk_text = " ".join([text for _, _, text in chunk])

                # Analyze the emotions in the chunk of text using the emotion analyzer
                chunk_emotions = emotion_analyzer(chunk_text)

                # Iterate over the emotions
                for emotion, (start_time, end_time, text) in zip(chunk_emotions, chunk):
                    # Add the start time to the emotion dictionary
                    emotion["start"] = start_time

                    # Add the end time to the emotion dictionary
                    emotion["end"] = end_time

                    # Add the text to the emotion dictionary
                    emotion["text"] = text

                # Add the emotion analysis results to the list of emotions
                emotions.extend(chunk_emotions)

            # Save the emotion analysis results to a file
            save_emotion_analysis(emotions, emotion_path)

        st.write("Emotion analysis complete!")  # Display a message
    else:  # If the emotion analysis file exists

        # Load the emotion analysis results from the file
        emotions = load_emotion_analysis(emotion_path)

        st.write("Using existing emotion analysis data.")  # Display a message
    return emotions
