import os
import torch
import sys
from faster_whisper import WhisperModel
import json

def transcribe_audio(audio_path, transcript_path):
    model = None
    try:
        if not os.path.exists(transcript_path):
            # If CUDA is available, use it; otherwise, use the CPU
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            
            print(f"Initializing Audio Transcription Model on: {device_str}")

            model = WhisperModel(
                "base.en",
                device=device_str,
                compute_type="float16",
            )
            
            # Transcribe the audio file
            print("Starting Audio Transcription Now...")
            segments, _ = model.transcribe(
                audio_path,
                beam_size=5,
                language="en",
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                word_timestamps=True
            )

            # Convert the segments generator to a list
            segment_list = list(segments)

            # Extract the start, end, text and words from each segment
            transcription = []
            for seg in segment_list:
                # Dictionary to store individual words
                words = []

                # Loop through seg.words and extract start, end, word
                for word in seg.words:
                    words.append({
                        "start": float(word.start),
                        "end": float(word.end),
                        "word": str(word.word)
                    })
                transcription.append({
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": str(seg.text),
                    "words": words
                })

            # Write the JSON file in the parent process
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump(transcription, f, indent=4, ensure_ascii=False)

            return transcription
          
        else:  
            # If The Transcription Already Exists
            print("The Transcription Already Exists; Skipping Transcription Process...")

            # Read the JSON file and return the data
            with open(transcript_path, "r", encoding="utf-8") as f:
                return json.load(f)
            
    except Exception as e:
        print(f"Fatal Error in transcribe_audio: {str(e)}")
        print("Stack trace:", sys.exc_info())
        return []