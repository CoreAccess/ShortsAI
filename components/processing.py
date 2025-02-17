import os
import shutil
from components.helpers import get_file_hash, extractAudio
from components.sentiment_analysis import analyze_emotions
from components.subtitles import write_ass, burn_subtitles
from components.transcriptions import transcribe_audio
from components.conversations import conversation_detection
from components.edits import detect_face_and_crop
import json

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

        progress_dict[filename] = {"progress": 5, "error": False}

        # Generate a hash for the video file
        file_hash = get_file_hash(temp_file_path)
        audio_path = os.path.join(temp_dir, f"{file_hash}_audio.wav")
        transcript_path = os.path.join(temp_dir, f"{file_hash}_transcript.json")

        print("Starting The Audio Extraction Process...")
        progress_dict[filename] = {"progress": 10, "error": False}

        # Extract The Audio From The Video (temp_file_path)
        if not os.path.exists(audio_path):
            audio_path = extractAudio(temp_file_path, audio_path)
                
            if audio_path is None:
                raise Exception("Audio Extraction Has Failed!")
            else:
                print("Audio Extraction Was a Success...")
        else:
            print("The Audio File Already Exists; Skipping Extraction...")

        progress_dict[filename] = {"progress": 25, "error": False}

        print("Preparing For Audio Transcription...")

        transcription = transcribe_audio(audio_path, transcript_path)

        progress_dict[filename] = {"progress": 40, "error": False}
        
        print("Starting Conversation Detection Process...")
        conversations = conversation_detection(transcription)

        # Save the conversation segments to a file for debugging as json
        conversation_path = os.path.join(temp_dir, f"conversations.json")
        with open(conversation_path, 'w') as f:
            json.dump(conversations, f, indent=4)

        print("Finished Conversation Detection...")

        print("Starting Sentiment Analysis Process...")
        try:
            emotions = analyze_emotions(conversations)
            if not emotions:
                raise Exception("Sentiment Analysis Failed...")
        except Exception as e:
            print(f"Error in Sentiment Analysis: {str(e)}")
            progress_dict[filename] = {"progress": 0, "error": True}
            raise

        # Save the conversation segments to a file for debugging as json
        emotion_path = os.path.join(temp_dir, f"emotions.json")
        with open(emotion_path, 'w') as f:
            json.dump(emotions, f, indent=4)

        print("Finished Sentiment Analysis Process...")

        progress_dict[filename] = {"progress": 60, "error": False}

        # For each emotions segment, only keep the ones with a emotion_score > 0.5
        interesting_segments = [seg for seg in emotions if seg["emotion_score"] > 0.5]

        # Initialize all transcript entries with conversation_id 0
        for entry in transcription:
            entry["conversation_id"] = "0"
            entry_text = entry["text"].lower()
            
            # Check if this transcript entry matches any interesting segment
            for idx, segment in enumerate(interesting_segments, 1):
                segment_text = segment["text"].lower()
                if entry_text in segment_text or segment_text in entry_text:
                    entry["conversation_id"] = str(idx)
                    break  # Stop checking once we find a match

        new_transcript_path = os.path.join(temp_dir, f"NEW-TRANSCRIPT.json")
        with open(new_transcript_path, 'w') as f:
            json.dump(transcription, f, indent=4)

        print(f"Updated transcript with {len(interesting_segments)} potential conversation segments")

        # Clamp this to 2 for now, remove it later on
        interesting_segments = interesting_segments[:1]  

        # Process each interesting segment
        for idx, segment in enumerate(interesting_segments, 1):
            start_time = segment["start"]
            end_time = segment["end"]

            progress_dict[filename] = {"progress": 65, "error": False}

            try:
                # Call detect_face_and_crop with just video_path, start_time, and end_time
                cropped_file = detect_face_and_crop(temp_file_path, start_time, end_time)
                if cropped_file:
                    print(f"Clip Was Extracted To: {cropped_file}")
                else:
                    print("Failed to extract clip")
                    continue
            except ValueError as e:
                print(f"No faces detected in the video segment: {str(e)}")
                continue

            # Debugging statement to check if the cropped file exists
            if os.path.exists(cropped_file):
                print(f"Cropped file exists: {cropped_file}")
            else:
                print(f"Cropped file does not exist: {cropped_file}")

            progress_dict[filename] = {"progress": 80, "error": False}

            # Create subtitles using only the transcription entries that match this conversation
            subtitles = []
            sentence_text = []
            current_sentence = 0
            last_end_time = None
            
            # Filter transcription entries that belong to this conversation
            conversation_segments = [seg for seg in transcription if seg["conversation_id"] == str(idx)]
            
            for segment in conversation_segments:
                # Loop through each word in the segment
                for word_info in segment["words"]:
                    current_word = word_info["word"].strip()
                    # Adjust timestamps relative to the clip's start time
                    word_start = word_info["start"] - start_time
                    word_end = word_info["end"] - start_time
                    
                    # Only include words that fall within the clip's time range
                    if word_start >= 0 and word_end <= (end_time - start_time):
                        # Detect sentence boundaries
                        ends_sentence = any(current_word.endswith(x) for x in ['.', '!', '?', '..."', '."', '!"', '?"'])
                        
                        # Check for significant pauses that might indicate sentence breaks
                        if last_end_time is not None:
                            time_gap = word_start - last_end_time
                            if time_gap > 1.5:
                                current_sentence += 1
                                sentence_text = []
                        
                        sentence_text.append(current_word)
                        subtitles.append((word_start, word_end, current_word, current_sentence))
                        
                        if ends_sentence:
                            current_sentence += 1
                            sentence_text = []
                        
                        last_end_time = word_end

            ass_file = os.path.join(temp_dir, f"{file_hash}_subtitles_{start_time:.2f}_{end_time:.2f}.ass")

            # Only proceed with subtitle burning if we have a cropped file
            if os.path.exists(cropped_file):
                print("Starting Subtitle Generation...")
                write_ass(subtitles, ass_file, cropped_file)

                subtitled_file = os.path.join(temp_dir, f"{file_hash}_dramatic_clip_with_subtitles_{start_time:.2f}_{end_time:.2f}.mp4")
                if not os.path.exists(subtitled_file):
                    burn_subtitles(cropped_file, ass_file, subtitled_file)
                    print(f"Generated Clip With Subtitles At: {subtitled_file}")
                else:
                    print(f"Existing Clip Already Exists Using: {subtitled_file}")

                # Only try to copy if the subtitled file was created successfully
                if os.path.exists(subtitled_file):
                    print(f"Final Clip Ready For Viewing At: {subtitled_file}")
                    progress_dict[filename] = {"progress": 95, "error": False}

                    # Move the final video to finished_videos folder
                    final_video_name = f"{os.path.splitext(filename)[0]}_short_{start_time:.2f}_{end_time:.2f}.mp4"
                    final_video_path = os.path.join(finished_dir, final_video_name)
                    try:
                        shutil.copy2(subtitled_file, final_video_path)
                        print(f"Moved final video to: {final_video_path}")
                    except Exception as e:
                        print(f"Error copying final video: {str(e)}")
                        progress_dict[filename] = {"progress": 0, "error": True}
                        continue
                else:
                    print("Subtitled file was not created successfully")
                    progress_dict[filename] = {"progress": 0, "error": True}
                    continue
            else:
                print(f"Skipping subtitle generation - cropped file does not exist: {cropped_file}")
                progress_dict[filename] = {"progress": 0, "error": True}
                continue

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
        '''
        # Only clean up temp files after all segments are processed
        if progress_dict[filename]["progress"] == 100:
            print("Processing complete, cleaning up temporary files...")
            # Loop through and delete all files in the temp_files folder
            for file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        print(f"Deleted temporary file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file: {file_path}")
                    print(e)
        '''