import hashlib
import cv2

# ---------------------------------------------------------------
# Compute MD5 hash of a file.
# ---------------------------------------------------------------


def get_file_hash(file_path):
    # Create an MD5 hash object
    hasher = hashlib.md5()

    # Open the file in binary read mode
    with open(file_path, 'rb') as f:
        # Read the entire file into a buffer
        buf = f.read()

        # Update the hash object with the file's content
        hasher.update(buf)

    # Return the hexadecimal representation of the hash
    return hasher.hexdigest()

# ---------------------------------------------------------------
# Break the transcription segments into chunks.
# ---------------------------------------------------------------


def chunk_text_with_timestamps(transcription_segments, max_length=192):
    # Initialize an empty list to store the chunks
    chunks = []

    # Initialize an empty list to store the current chunk
    current_chunk = []

    # Initialize the current length of the chunk to 0
    current_length = 0

    # Iterate over each transcription segment
    for segment in transcription_segments:
        # Get the start time of the segment
        start_time = segment["timestamp"][0]

        # Get the end time of the segment
        end_time = segment["timestamp"][1]

        # Get the text of the segment
        text = segment["text"]

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

            # Update the current length
            current_length += len(text.split())

    # If there's a remaining chunk, append it to the list of chunks
    if current_chunk:
        chunks.append(current_chunk)

    # Return the list of text chunks
    return chunks

# ---------------------------------------------------------------
# Crop a single video frame to a 9:16 aspect ratio.
# ---------------------------------------------------------------


def crop_frame(frame, left, right, target_width, width, height):

    # Crop the frame horizontally
    cropped_frame = frame[:, int(left):int(right)]

    # Resize the cropped frame to the target dimensions (1080x1920)
    return cv2.resize(cropped_frame, (1080, 1920))

# ---------------------------------------------------------------
# Load and parse transcription segments from a transcript file.
# ---------------------------------------------------------------


def load_transcription_segments(transcript_path):

    # Initialize an empty list to store transcription segments
    transcription_segments = []

    # Open the transcript file in read mode
    with open(transcript_path, 'r', encoding='utf-8') as f:
        # Iterate over each line in the file
        for line in f:
            # Check if the line is not empty
            if line.strip():
                parts = line.strip().split(']')
                if len(parts) == 2:
                    timestamp, text = parts
                    start_time, end_time = map(
                        float, timestamp[1:].split(' - '))
                    transcription_segments.append(
                        {"timestamp": [start_time, end_time], "text": text})
    return transcription_segments

# ---------------------------------------------------------------
# Write emotion analysis data to file.
# ---------------------------------------------------------------


def save_emotion_analysis(emotions, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for emotion in emotions:
            f.write(
                f"{emotion['start']} - {emotion['end']} - {emotion['label']} - {emotion['score']} - {emotion['text']}\n")

# ---------------------------------------------------------------
# Read previously saved emotion analysis results.
# ---------------------------------------------------------------


def load_emotion_analysis(file_path):
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

# ---------------------------------------------------------------
# Convert seconds (float) to SRT timestamp format (HH:MM:SS,ms).
# ---------------------------------------------------------------


def format_timestamp(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    millis = int(round((secs - int(secs)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{int(secs):02d},{millis:03d}"
