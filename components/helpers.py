import hashlib
import cv2
import json
import subprocess

# ----------------------------------------------------------------------------
# Return the duration of a video using ffprobe.
# ----------------------------------------------------------------------------


def get_video_duration(video_path):
    """Return video duration in seconds using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'format=duration',
        '-of', 'json',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("ffprobe error: " + result.stderr)
    info = json.loads(result.stdout)
    return float(info['format']['duration'])

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

# Helper function to find the start of a sentence

def find_sentence_start(transcription_segments, start_index):
    """
    Find the start of a complete sentence before or at start_index.
    Returns the index where the sentence starts.
    """
    # If we're already at the start of a sentence, return current index
    if start_index == 0 or not transcription_segments[start_index-1]["text"].strip():
        return start_index

    # Look backwards for the end of the previous sentence
    for i in range(start_index - 1, -1, -1):
        current_text = transcription_segments[i]["text"].strip()
        next_text = transcription_segments[i + 1]["text"].strip()
        
        # Check if current segment ends with sentence-ending punctuation
        if current_text.endswith(('.', '!', '?')):
            # Return the start of the next segment
            return i + 1
        
        # Check for natural pauses (commas, etc) if we've gone back more than 3 segments
        if i < start_index - 3 and current_text.endswith(','):
            return i + 1
    
    # If no good break point found, return original start_index
    return start_index

# Helper function to find the end of a sentence

def find_sentence_end(transcription_segments, start_index):
    """
    Find the end of a complete sentence starting from start_index.
    Returns the index where the sentence ends.
    """
    max_segments = len(transcription_segments)
    current_index = start_index
    
    while current_index < max_segments:
        current_text = transcription_segments[current_index]["text"].strip()
        
        # Check for sentence-ending punctuation
        if current_text.endswith(('.', '!', '?')):
            return current_index
            
        # If we've gone too far without finding a sentence end,
        # look for a natural pause point
        if current_index > start_index + 5:
            if current_text.endswith(','):
                return current_index
        
        current_index += 1
    
    # If no sentence end found, return the last available index
    return min(start_index + 5, max_segments - 1)

# Add a helper function at the top (after the imports)
def normalize_path(path):
    return path.replace('\\', '/')