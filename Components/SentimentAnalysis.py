# Import the pipeline function from the transformers library
from transformers import pipeline


def analyze_emotions(transcription_segments, emotion_path, st, torch, device_id):
    """
    Analyze emotions from the transcription segments using an emotion classifier.
    """
    import os  # Import the os module for interacting with the operating system
    # Import helper functions
    from Components.Helpers import chunk_text_with_timestamps, save_emotion_analysis, load_emotion_analysis
    if not os.path.exists(emotion_path):
        # Check if the emotion analysis file does not exist
        with st.spinner("Analyzing emotions..."):
            # Display a spinner in the streamlit app while analyzing emotions
            emotion_analyzer = pipeline(
                "text-classification",
                model="michellejieli/emotion_text_classifier",
                device=device_id
            )
            # Create a pipeline for text classification using the specified model and device
            transcription_chunks = chunk_text_with_timestamps(
                transcription_segments)
            # Chunk the transcription segments into smaller pieces with timestamps
            emotions = []
            # Initialize an empty list to store the emotion analysis results
            for chunk in transcription_chunks:
                # Iterate over each chunk of transcription segments
                chunk_text = " ".join([text for _, _, text in chunk])
                # Extract the text from the chunk and join it into a single string
                chunk_emotions = emotion_analyzer(chunk_text)
                # Analyze the emotions in the chunk of text using the emotion analyzer
                for emotion, (start_time, end_time, text) in zip(chunk_emotions, chunk):
                    # Iterate over the emotions and corresponding timestamps and text in the chunk
                    emotion["start"] = start_time
                    # Add the start time to the emotion dictionary
                    emotion["end"] = end_time
                    # Add the end time to the emotion dictionary
                    emotion["text"] = text
                    # Add the text to the emotion dictionary
                emotions.extend(chunk_emotions)
                # Add the emotion analysis results to the list of emotions
            save_emotion_analysis(emotions, emotion_path)
            # Save the emotion analysis results to a file
        st.write("Emotion analysis complete!")
        # Write a message to the streamlit app indicating that the emotion analysis is complete
    else:
        # If the emotion analysis file exists
        from Components.Helpers import load_emotion_analysis
        # Import the load_emotion_analysis function
        emotions = load_emotion_analysis(emotion_path)
        # Load the emotion analysis results from the file
        st.write("Using existing emotion analysis data.")
        # Write a message to the streamlit app indicating that existing emotion analysis data is being used
    return emotions
