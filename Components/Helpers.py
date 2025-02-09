def get_file_hash(file_path):
    """
    Compute MD5 hash of a file.
    This helps in reusing extracted files by ensuring we process unique inputs only.
    """
    import hashlib
    hasher = hashlib.md5()  # Create an MD5 hash object
    with open(file_path, 'rb') as f:  # Open the file in binary read mode
        buf = f.read()  # Read the entire file into a buffer
        hasher.update(buf)  # Update the hash object with the file's content
    return hasher.hexdigest()  # Return the hexadecimal representation of the hash


def chunk_text_with_timestamps(transcription_segments, max_length=192):
    """
    Break the transcription segments into chunks.
    Each chunk's combined word count should not exceed max_length.
    This helps in batching text for emotion analysis.
    """
    chunks = []  # Initialize an empty list to store the chunks
    current_chunk = []  # Initialize an empty list to store the current chunk
    current_length = 0  # Initialize the current length of the chunk to 0

    for segment in transcription_segments:  # Iterate over each transcription segment
        # Get the start time of the segment
        start_time = segment["timestamp"][0]
        end_time = segment["timestamp"][1]  # Get the end time of the segment
        text = segment["text"]  # Get the text of the segment

        # Check if adding the current segment exceeds the maximum length
        if current_length + len(text.split()) > max_length:
            # Append the current chunk to the list of chunks
            chunks.append(current_chunk)
            # Start a new chunk with the current segment
            current_chunk = [(start_time, end_time, text)]
            # Update the current length with the length of the current segment
            current_length = len(text.split())
        else:
            # Add the current segment to the current chunk
            current_chunk.append((start_time, end_time, text))
            current_length += len(text.split())  # Update the current length

    if current_chunk:  # If there's a remaining chunk, append it to the list of chunks
        chunks.append(current_chunk)

    return chunks  # Return the list of text chunks


def crop_frame(frame, left, right, target_width, width, height):
    """
    Crop a single video frame to a 9:16 aspect ratio.
    The cropping is done horizontally based on the provided boundaries.
    """
    import cv2
    # Crop the frame horizontally
    cropped_frame = frame[:, int(left):int(right)]
    # Resize the cropped frame to the target dimensions (1080x1920)
    return cv2.resize(cropped_frame, (1080, 1920))


def load_transcription_segments(transcript_path):
    """
    Load and parse transcription segments from a transcript file.
    Each line is expected in the format: "[start - end] text"
    """
    transcription_segments = []  # Initialize an empty list to store transcription segments
    with open(transcript_path, 'r', encoding='utf-8') as f:  # Open the transcript file
        for line in f:  # Iterate over each line in the file
            if line.strip():  # Check if the line is not empty
                parts = line.strip().split(']')
                if len(parts) == 2:
                    timestamp, text = parts
                    start_time, end_time = map(
                        float, timestamp[1:].split(' - '))
                    transcription_segments.append(
                        {"timestamp": [start_time, end_time], "text": text})
    return transcription_segments


def save_emotion_analysis(emotions, file_path):
    """
    Write emotion analysis data to file.
    Each line records the start and end timestamps along with emotion label, score, and the related text.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for emotion in emotions:
            f.write(
                f"{emotion['start']} - {emotion['end']} - {emotion['label']} - {emotion['score']} - {emotion['text']}\n")


def load_emotion_analysis(file_path):
    """
    Read previously saved emotion analysis results.
    Expects each line in the saved file to be formatted with dash-separated fields.
    """
    emotions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            start, end, label, score, text = line.strip().split(' - ')
            emotions.append({
                'start': float(start),
                'end': float(end),
                'label': label,
                'score': float(score),
                'text': text
            })
    return emotions


def format_timestamp(seconds):
    """
    Convert seconds (float) to SRT timestamp format (HH:MM:SS,ms).
    """
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    millis = int(round((secs - int(secs)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{int(secs):02d},{millis:03d}"
