import os
import shutil  # Import shutil for file operations
from components.edits import extractAudio, detect_face_and_crop
from components.helpers import get_file_hash, load_transcription_segments
from components.transcriptions import transcribe_audio
from components.sentiment_analysis import analyze_emotions
from components.subtitles import write_srt, burn_subtitles
from moviepy.video.io.VideoFileClip import VideoFileClip

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
            temp_file_path = shutil.copy(video_path, temp_file_path)
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
        transcription_success = False
        max_retries = 3
        retry_count = 0
        
        while not transcription_success and retry_count < max_retries:
            try:
                transcription_segments = transcribe_audio(audio_path, transcript_path)
                if not transcription_segments and os.path.exists(transcript_path):
                    # If transcription file exists but segments are empty, try to load it
                    transcription_segments = load_transcription_segments(transcript_path)
                
                if not transcription_segments:
                    retry_count += 1
                    print(f"Transcription attempt {retry_count} failed, {'retrying' if retry_count < max_retries else 'giving up'}...")
                    continue
                
                transcription_success = True
                print("Transcription completed successfully")
            except Exception as e:
                print(f"Error in transcription attempt {retry_count + 1}: {str(e)}")
                retry_count += 1
                if retry_count >= max_retries:
                    progress_dict[filename] = {"progress": 0, "error": True}
                    raise Exception(f"Transcription failed after {max_retries} attempts")
        
        progress_dict[filename] = {"progress": 40, "error": False}

        print("Starting Sentiment Analysis Process...")
        try:
            emotions = analyze_emotions(transcription_segments, emotion_path)
            if not emotions:
                raise Exception("Sentiment analysis failed or returned empty")
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            progress_dict[filename] = {"progress": 0, "error": True}
            raise
        progress_dict[filename] = {"progress": 55, "error": False}

        # Rest of the processing code...
        dramatic_segments = [segment for segment in emotions if segment['label'] in [
            'anger', 'fear', 'sadness']]

        if dramatic_segments:
            progress_dict[filename] = {"progress": 60, "error": False}
            base_start = float(dramatic_segments[0]['start'])

            for segment in transcription_segments:
                if segment["timestamp"][1] >= base_start:
                    start_time = segment["timestamp"][0]
                    break

            max_duration = 59.0
            segments_in_range = [
                segment for segment in transcription_segments
                if segment["timestamp"][0] >= start_time and segment["timestamp"][1] <= start_time + max_duration
            ]
            candidate_end = start_time
            for segment in segments_in_range:
                text = segment["text"].strip()
                if text.endswith((".", "?", "!")):
                    candidate_end = segment["timestamp"][1]
            if candidate_end == start_time and segments_in_range:
                candidate_end = segments_in_range[-1]["timestamp"][1]
            end_time = candidate_end

            print(f"Extracting Clip From {start_time:.2f}s To {end_time:.2f}s.")
            progress_dict[filename] = {"progress": 65, "error": False}

            cropped_file = f"{temp_dir}/{file_hash}_dramatic_clip.mp4"
            subtitled_file = f"{temp_dir}/{file_hash}_dramatic_clip_with_subtitles.mp4"

            if not os.path.exists(cropped_file):
                detect_face_and_crop(temp_file_path, cropped_file, start_time, end_time)
                print(f"Clip Was Extracted To: {cropped_file}")
            else:
                print(f"Clip Already Exists Using: {cropped_file}")
            progress_dict[filename] = {"progress": 80, "error": False}

            subtitles = [
                (segment["timestamp"][0] - start_time, segment["timestamp"][1] - start_time,
                segment["text"].replace('\u266a', '*'))
                for segment in transcription_segments
                if segment["timestamp"][0] >= start_time and segment["timestamp"][1] <= end_time
            ]

            srt_file = f"{temp_dir}/{file_hash}_subtitles.srt"
            print("Starting Subtitle Generation...")
            write_srt(subtitles, srt_file)
            progress_dict[filename] = {"progress": 90, "error": False}

            if not os.path.exists(subtitled_file):
                burn_subtitles(cropped_file, srt_file, subtitled_file)
                print(f"Generated Clip With Subtitles At: {subtitled_file}")
            else:
                print(f"Existing Clip Already Exists Using: {subtitled_file}")

            print(f"Final Clip Ready For Viewing At: {subtitled_file}")
            progress_dict[filename] = {"progress": 95, "error": False}

            # Move the final video to finished_videos folder
            final_video_name = f"{os.path.splitext(filename)[0]}_short.mp4"
            final_video_path = os.path.join(finished_dir, final_video_name)
            shutil.copy2(subtitled_file, final_video_path)
            print(f"Moved final video to: {final_video_path}")

            # Clean up temp files
            temp_files = [
                audio_path,
                transcript_path,
                emotion_path,
                cropped_file,
                srt_file,
                subtitled_file
            ]
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"Deleted temporary file: {temp_file}")

            # Delete the original video from uploads
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"Deleted original video from uploads: {video_path}")

            progress_dict[filename] = {"progress": 100, "error": False}
        else:
            print("No Dramatic Segments Detected...")
            progress_dict[filename] = {"progress": 0, "error": True}

        print("Processing Completed Successfully! Go Watch The Clip!")

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
            except Exception as e:
                print(f"Error cleaning up temp file: {str(e)}")