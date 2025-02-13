import os
import torch
import sys
from faster_whisper import WhisperModel
from components.helpers import load_transcription_segments

def transcribe_audio(audio_path, transcript_path):
    model = None
    try:
        if not os.path.exists(transcript_path):
            # Initialize model
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            
            print(f"Initializing model on {device_str}...")
            model = WhisperModel(
                "base.en",
                device=device_str,
            )
            
            # Transcribe the audio file
            print("Transcribing audio file...")
            segments, _ = model.transcribe(
                audio_path,
                beam_size=4,
                language="en",
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                word_timestamps=True
            )
            
            # Combine transcription results
            transcription_segments = []
            for seg in segments:
                for word in seg.words:
                    start, end, text = word.start, word.end, word.word.replace('\u266a', '*Music*')
                    transcription_segments.append({
                        "timestamp": [start, end],  # Keep timestamps in seconds
                        "text": text
                    })

            # Save final transcription
            print("Saving final transcription...")
            with open(transcript_path, 'w', encoding='utf-8') as f:
                for segment in transcription_segments:
                    start, end = segment["timestamp"]
                    text = segment["text"]
                    f.write(f"[{start:.3f} - {end:.3f}] {text}\n")
            
            print("Transcription completed successfully")
            return transcription_segments
            
        else:  # Transcript exists
            print("Using existing transcription...")
            return load_transcription_segments(transcript_path)
            
    except Exception as e:
        print(f"Fatal error in transcribe_audio: {str(e)}")
        print("Stack trace:", sys.exc_info())
        return []
        
    finally:
        # Clean up resources
        try:
            if model is not None:
                del model
        except Exception as e:
            print(f"Error during final cleanup: {str(e)}")
