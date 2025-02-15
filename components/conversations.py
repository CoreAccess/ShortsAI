def has_excessive_repetition(text, threshold=4):
    """Check if text contains excessive word repetition."""
    words = text.lower().split()
    if len(words) < threshold:
        return False
        
    # Count consecutive repeated words
    max_repetitions = 1
    current_repetitions = 1
    prev_word = words[0]
    
    for word in words[1:]:
        if word == prev_word:
            current_repetitions += 1
            max_repetitions = max(max_repetitions, current_repetitions)
        else:
            current_repetitions = 1
        prev_word = word
    
    return max_repetitions >= threshold

def conversation_detection(transcription):
    MAX_DURATION = 59.5
    MIN_DURATION = 30.0  # Minimum duration to consider a segment interesting
    SILENCE_THRESHOLD = 5.0  # Maximum allowed gap between sentences in seconds
    segments = []
    
    def get_segment_text(start_idx, end_idx):
        return ' '.join(seg["text"] for seg in transcription[start_idx:end_idx + 1])
    
    def find_natural_end(start_idx, max_end_idx):
        """Find a natural ending point that doesn't cut off mid-sentence"""
        for i in range(max_end_idx, start_idx, -1):
            text = transcription[i]["text"].strip()
            # Check for sentence-ending punctuation
            if text.endswith(('.', '!', '?')):
                return i
        # If no natural end found, return the latest possible index
        return max_end_idx
    
    i = 0
    temp_segments = []
    while i < len(transcription):
        start_time = transcription[i]["start"]
        
        # Find the furthest we can go within MAX_DURATION
        j = i
        while j < len(transcription) and transcription[j]["end"] - start_time <= MAX_DURATION:
            j += 1
        j = min(j - 1, len(transcription) - 1)  # Ensure we don't go past the end
        
        # Find a natural endpoint
        end_idx = find_natural_end(i, j)
        end_time = transcription[end_idx]["end"]
        
        # Check for gaps between sentences
        valid_segment = True
        for k in range(i, end_idx):
            if k + 1 < len(transcription):
                gap = transcription[k + 1]["start"] - transcription[k]["end"]
                if gap > SILENCE_THRESHOLD:
                    valid_segment = False
                    break
        
        # Get segment text and check for excessive repetition
        segment_text = get_segment_text(i, end_idx)
        if has_excessive_repetition(segment_text):
            valid_segment = False
        
        # Check if segment meets minimum duration and has no large gaps
        if valid_segment and end_time - start_time >= MIN_DURATION:
            temp_segments.append((start_time, end_time, segment_text))
        
        # Move to start of next potential segment
        i = end_idx + 1
    
    # Remove first two and last two segments if we have enough segments
    if len(temp_segments) > 4:
        segments = temp_segments[2:-2]
    else:
        # If we have 4 or fewer segments, just return middle ones if possible
        if len(temp_segments) > 2:
            segments = temp_segments[1:-1]
        else:
            segments = temp_segments  # Return all segments if we have 2 or fewer
            
    return segments
