import os
import shutil  # Import shutil for file operations
from components.edits import extractAudio, detect_face_and_crop
from components.helpers import get_file_hash, load_transcription_segments, find_best_chunks, find_sentence_start, find_sentence_end
from components.transcriptions import transcribe_audio
from components.sentiment_analysis import analyze_emotions
from components.subtitles import write_ass, burn_subtitles
from moviepy.video.io.VideoFileClip import VideoFileClip
import sys

def convert_to_sentence_level(word_segments):
    sentence_segments = []
    current_sentence = []
    start_time = None
    for word in word_segments:
        if start_time is None:
            start_time = word["timestamp"][0]
        current_sentence.append(word["text"])
        if word["text"].endswith(('.', '?', '!')):
            end_time = word["timestamp"][1]
            sentence_segments.append({
                "timestamp": [start_time, end_time],
                "text": ' '.join(current_sentence)
            })
            current_sentence = []
            start_time = None
    if current_sentence:
        end_time = word_segments[-1]["timestamp"][1]
        sentence_segments.append({
            "timestamp": [start_time, end_time],
            "text": ' '.join(current_sentence)
        })
    return sentence_segments

def process_video(video_path, progress_dict, temp_dir, finished_dir):
    filename = os.path.basename(video_path)
    progress_dict[filename] = {"progress": 0, "error": False}
    temp_file_path = None
    
    try:
        # Create required directories if they don't exist
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(finished_dir, exist_ok=True)

        # Check if the video already exists in temp_files
        temp_file_path = os.path.join(temp_dir, os.path.basename(video_path))
        if not os.path.exists(temp_file_path):
            shutil.copy(video_path, temp_file_path)
        else:
            print(f"Video file already exists in temp_files, using existing file: {temp_file_path}")

        print("Video Processing Has Begun...")
        progress_dict[filename] = {"progress": 5, "error": False}

        # Generate a hash for the video file
        file_hash = get_file_hash(temp_file_path)
        audio_path = f"{temp_dir}/{file_hash}_audio.wav"
        transcript_path = f"{temp_dir}/{file_hash}_transcript.txt"
        emotion_path = f"{temp_dir}/{file_hash}_emotions.txt"

        print("Starting The Audio Processing...")
        progress_dict[filename] = {"progress": 10, "error": False}

        try:
            # Audio extraction with error handling
            if not os.path.exists(audio_path):
                with VideoFileClip(temp_file_path) as video:
                    audio_path = extractAudio(temp_file_path, audio_path)
                if audio_path is None:
                    raise Exception("Audio extraction failed")
                print("Audio Processing Was a Success...")
            else:
                print("Audio File Already Exists; Skipping Extraction...")
        except Exception as e:
            print(f"Error in audio extraction: {str(e)}")
            progress_dict[filename] = {"progress": 0, "error": True}
            raise

        progress_dict[filename] = {"progress": 25, "error": False}

        print("Starting Audio Transcription Process...")

        word_segments = transcribe_audio(audio_path, transcript_path)
        word_segments = load_transcription_segments(transcript_path)

        # Convert word-level transcription to sentence-level transcription
        sentence_segments = convert_to_sentence_level(word_segments)

        progress_dict[filename] = {"progress": 40, "error": False}

        print("Starting Sentiment Analysis Process...")
        try:
            emotions = analyze_emotions(sentence_segments, emotion_path)
            if not emotions:
                raise Exception("Sentiment analysis failed or returned empty")
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            progress_dict[filename] = {"progress": 0, "error": True}
            raise

        progress_dict[filename] = {"progress": 60, "error": False}

        # Find the best segments that are close to 59 seconds with highest emotional scores
        best_chunks = find_best_chunks(emotions)
        if not best_chunks:
            print("No suitable segments found")
            progress_dict[filename] = {"progress": 0, "error": True}
            raise Exception("No suitable segments found")
        
        # print out the total best_chunks
        print(f"Total best_chunks: {len(best_chunks)}")

        # Process top 3 non-overlapping segments
        best_chunks = best_chunks[:1]

        cropped_file = None
        subtitled_file = None
        ass_file = None

        for chunk in best_chunks:
            start_index, end_index, start_time, end_time = chunk
            print(f"Processing chunk: Duration={end_time-start_time:.2f}s, Start={start_time:.2f}s, End={end_time:.2f}s")
            
            # Get word segments that fall within this chunk's time range, with a small buffer
            buffer = 0.1  # 100ms buffer to catch nearby words
            chunk_words = [
                segment for segment in word_segments
                if (segment["timestamp"][0] >= (start_time - buffer) and 
                    segment["timestamp"][1] <= (end_time + buffer))
            ]
            
            if not chunk_words:
                print(f"No word segments found for chunk {start_time:.2f}-{end_time:.2f}")
                continue

            print(f"Extracting Clip From {start_time:.2f}s To {end_time:.2f}s.")

            progress_dict[filename] = {"progress": 65, "error": False}

            cropped_file = f"{temp_dir}/{file_hash}_dramatic_clip_{start_time:.2f}_{end_time:.2f}.mp4"
            subtitled_file = f"{temp_dir}/{file_hash}_dramatic_clip_with_subtitles_{start_time:.2f}_{end_time:.2f}.mp4"

            if not os.path.exists(cropped_file):
                try:
                    detect_face_and_crop(temp_file_path, cropped_file, start_time, end_time)
                    print(f"Clip Was Extracted To: {cropped_file}")
                except ValueError as e:
                    print(f"No faces detected in the video segment: {str(e)}")
                    continue
            else: 
                print(f"Clip Already Exists Using: {cropped_file}")

            progress_dict[filename] = {"progress": 80, "error": False}

            # Create subtitles for the chunk using word-level segments
            subtitles = [
                (segment["timestamp"][0] - start_time, 
                 segment["timestamp"][1] - start_time,
                 segment["text"])
                for segment in chunk_words
            ]

            ass_file = f"{temp_dir}/{file_hash}_subtitles_{start_time:.2f}_{end_time:.2f}.ass"
            print("Starting Subtitle Generation...")
            write_ass(subtitles, ass_file)
            progress_dict[filename] = {"progress": 90, "error": False}

            if not os.path.exists(subtitled_file):
                burn_subtitles(cropped_file, ass_file, subtitled_file)
                print(f"Generated Clip With Subtitles At: {subtitled_file}")
            else:
                print(f"Existing Clip Already Exists Using: {subtitled_file}")

            print(f"Final Clip Ready For Viewing At: {subtitled_file}")
            progress_dict[filename] = {"progress": 95, "error": False}

            # Move the final video to finished_videos folder
            final_video_name = f"{os.path.splitext(filename)[0]}_short_{start_time:.2f}_{end_time:.2f}.mp4"
            final_video_path = os.path.join(finished_dir, final_video_name)
            shutil.copy2(subtitled_file, final_video_path)
            print(f"Moved final video to: {final_video_path}")

            # Clean up temp files after all clips are created
            temp_files = [audio_path, transcript_path, emotion_path, cropped_file, ass_file, subtitled_file]
            for temp_file in temp_files:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"Deleted temporary file: {temp_file}")

        # Delete the original video from uploads
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"Deleted original video from uploads: {video_path}")

        progress_dict[filename] = {"progress": 100, "error": False}

    except Exception as e:
        print(f"Fatal error in process_video: {str(e)}")
        progress_dict[filename] = {"progress": 0, "error": True}
        # Ensure the error is logged but doesn't crash the server
        import traceback
        traceback.print_exc()
    finally:
        # Clean up temporary files regardless of success or failure
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"Temporary File Has Been Deleted From: {temp_file_path}")
                print("Processing Completed Successfully! Go Watch The Clips!")
            except Exception as e:
                print(f"Error cleaning up temp file: {str(e)}")