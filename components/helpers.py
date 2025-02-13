import hashlib
import cv2
import json
import subprocess
import re
import sys

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
                    # Check for speaker information
                    if '(' in text and ')' in text:
                        speaker, text = text.split(')', 1)
                        speaker = speaker[1:]  # Remove leading '('
                    else:
                        speaker = 'unknown'
                    transcription_segments.append(
                        {"timestamp": [start_time, end_time], "text": text.strip(), "speaker": speaker})
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

# Helper function to find multiple 59s chunks with the most non-neutral tags

def find_best_chunks(emotions, max_duration=59.0, min_duration=25.0):
    best_segments = []
    used_intervals = set()  # Track used time intervals
    target_duration = 59.0  # Target duration in seconds

    def calculate_segment_score(start_idx, end_idx):
        """Calculate the score for a segment based on emotional intensity and duration."""
        duration = emotions[end_idx]['end'] - emotions[start_idx]['start']
        if duration < min_duration or duration > max_duration:
            return float('-inf')
            
        # Calculate emotional score (sum of all emotion scores in segment)
        emotional_score = sum(emotions[i]['score'] for i in range(start_idx, end_idx + 1))
        
        # Calculate duration score (1.0 when duration = target_duration, decreasing as it deviates)
        duration_score = 1.0 - (abs(target_duration - duration) / target_duration)
        
        # Weight emotional score more heavily but still consider duration
        normalized_emotional_score = emotional_score / (end_idx - start_idx + 1)
        
        # Combine scores with weights (70% emotional, 30% duration)
        final_score = (0.7 * normalized_emotional_score) + (0.3 * duration_score)
        
        return final_score

    def is_overlapping(start_time, end_time):
        """Check if a time interval overlaps with any used intervals."""
        return any(start_time < used_end and end_time > used_start 
                  for used_start, used_end in used_intervals)

    def find_next_word_start(end_idx):
        """Find the start time of the next word after end_idx"""
        if end_idx + 1 < len(emotions):
            return emotions[end_idx + 1]['start']
        return None

    # Find all valid segments and their scores
    for start_idx in range(len(emotions)):
        # Skip if this start point would overlap with used intervals
        start_time = emotions[start_idx]['start']
        if any(start_time < used_end for _, used_end in used_intervals):
            continue

        # Find the actual start of the sentence
        sentence_start_idx = find_sentence_start(emotions, start_idx)
        if sentence_start_idx != start_idx:
            start_time = emotions[sentence_start_idx]['start']
            start_idx = sentence_start_idx

        end_idx = start_idx
        while end_idx < len(emotions):
            current_duration = emotions[end_idx]['end'] - start_time
            
            # Break early if we've exceeded max duration
            if current_duration > max_duration:
                break
                
            if min_duration <= current_duration <= max_duration:
                # Find the best sentence ending point
                sentence_end_idx = find_sentence_end(emotions, end_idx)
                end_idx = min(sentence_end_idx, len(emotions) - 1)
                
                # Find the start time of the next word to use as buffer
                next_word_start = find_next_word_start(end_idx)
                end_time = next_word_start if next_word_start and (next_word_start - start_time) <= max_duration else emotions[end_idx]['end']
                
                # Verify the adjusted duration is still within bounds
                adjusted_duration = end_time - start_time
                if min_duration <= adjusted_duration <= max_duration and not is_overlapping(start_time, end_time):
                    score = calculate_segment_score(start_idx, end_idx)
                    segment_data = {
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': adjusted_duration,
                        'score': score,
                        'emotional_sum': sum(emotions[i]['score'] 
                                           for i in range(start_idx, end_idx + 1))
                    }
                    best_segments.append(segment_data)
            
            end_idx += 1

    # Sort segments by score
    best_segments.sort(key=lambda x: x['score'], reverse=True)
    
    # Select non-overlapping segments with highest scores
    final_segments = []
    for segment in best_segments:
        if not is_overlapping(segment['start_time'], segment['end_time']):
            final_segments.append((
                segment['start_idx'],
                segment['end_idx'],
                segment['start_time'],
                segment['end_time']
            ))
            used_intervals.add((segment['start_time'], segment['end_time']))

    return final_segments 