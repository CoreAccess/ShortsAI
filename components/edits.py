import cv2
import ffmpeg
import os
import torch
import subprocess
import threading
import queue
import concurrent.futures 
import time
import psutil
from collections import deque
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
import mediapipe as mp
import numpy as np
import librosa
import wave
import contextlib
from pydub import AudioSegment

# Check if CUDA is available
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
face_detector_mp = None

# Global frames storage for face tracking
Frames = []

def prepare_frame(frame):
    """Convert frame to RGB for PyTorch processing"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if use_cuda:
        # Clear CUDA cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return frame_rgb 

# ----------------------------------------------------------------------------
# Extract audio from a video file
# ----------------------------------------------------------------------------


def extractAudio(video_path, audio_path):
    try:
        temp_files_dir = os.path.dirname(audio_path)
        print(f"Extracting audio to temp_files directory: {temp_files_dir}")
        
        audio = AudioSegment.from_file(video_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(audio_path, format="wav")
        print(f"Audio extracted successfully to: {audio_path}")
        return audio_path
    except Exception as e:
        print(f"An Error Occurred While Extracting Audio: {e}")
        return None

def process_audio_frame(audio_data, sample_rate=16000, frame_duration_ms=30):
    n = int(sample_rate * frame_duration_ms / 1000) * 2
    offset = 0
    while offset + n <= len(audio_data):
        frame = audio_data[offset:offset + n]
        offset += n
        yield frame

class FaceTracker:
    def __init__(self, frame_width, frame_height, vertical_width):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.vertical_width = vertical_width
        self.history_size = 45  # Increased for smoother transitions (1.5 seconds at 30fps)
        self.position_history = deque(maxlen=self.history_size)
        self.face_history = deque(maxlen=self.history_size)
        self.no_face_counter = 0
        self.max_no_face_frames = 60  # 2 seconds at 30fps
        self.last_valid_position = None
        self.center_position = (frame_width - vertical_width) // 2
        self.confidence_threshold = 0.85  # Increased confidence threshold
        self.movement_smoothing = 0.7  # Higher value = smoother movement
        self.speaker_bonus = 1.5  # Multiplier for speaker detection score

    def update(self, faces, is_speaking=False):
        if not faces:
            self.no_face_counter += 1
            if self.no_face_counter >= self.max_no_face_frames:
                if self.last_valid_position is not None:
                    # More gradual transition to center
                    alpha = min(1.0, (self.no_face_counter - self.max_no_face_frames) / 45)
                    position = int(self.last_valid_position * (1 - alpha) + self.center_position * alpha)
                else:
                    position = self.center_position
                self.position_history.append(position)
                return position
            elif self.last_valid_position is not None:
                self.position_history.append(self.last_valid_position)
                return self.last_valid_position
            else:
                self.position_history.append(self.center_position)
                return self.center_position

        self.no_face_counter = 0

        # Find the most prominent face with speaker detection
        best_face = None
        best_score = -1

        for face in faces:
            x, y, w, h = face
            center_x = x + w/2
            
            # Size score (prefer larger faces)
            size_score = (w * h) / (self.frame_width * self.frame_height)
            
            # Center score (prefer faces closer to center)
            center_offset = abs(center_x - self.frame_width/2)
            center_score = 1 - (center_offset / (self.frame_width/2))
            
            # Vertical position score (prefer faces in upper third)
            vertical_score = 1 - (y / self.frame_height)
            
            # Combine scores with weights
            score = (
                size_score * 0.4 +
                center_score * 0.4 +
                vertical_score * 0.2
            )
            
            # Apply speaker bonus if detected
            if is_speaking:
                score *= self.speaker_bonus
            
            if score > best_score:
                best_score = score
                best_face = face

        if best_score < 0.3:  # Strict minimum score threshold
            if self.last_valid_position is not None:
                return self.last_valid_position
            return self.center_position

        x, y, w, h = best_face
        face_center = x + w/2

        # Calculate crop position ensuring face is well within frame
        margin = w * 0.3  # 30% margin on each side of face
        crop_x = max(0, min(self.frame_width - self.vertical_width,
                           face_center - self.vertical_width/2))
        
        # Ensure face isn't too close to crop edges
        if x - margin < crop_x:
            crop_x = max(0, x - margin)
        elif (x + w + margin) > (crop_x + self.vertical_width):
            crop_x = min(self.frame_width - self.vertical_width,
                        x + w + margin - self.vertical_width)
        
        # Apply smoothing using weighted moving average
        self.position_history.append(crop_x)
        weights = np.exp(np.linspace(-1, 0, len(self.position_history)))
        weights /= weights.sum()
        smoothed_position = int(np.average(self.position_history, weights=weights))
        
        # Additional boundary check
        smoothed_position = max(0, min(self.frame_width - self.vertical_width, smoothed_position))
        
        self.last_valid_position = smoothed_position
        return smoothed_position

def detect_faces_and_speakers(input_video_path, output_video_path, frame_interval=2, buffer_time=1.0):
    """
    Enhanced face detection with better tracking and speaker detection.
    Works with pre-trimmed video clip, accounting for buffer time.
    """
    global Frames, face_detector_mp
    Frames = []
    
    print("\n=== Starting Face Detection Pipeline ===")
    print(f"Processing trimmed clip: {input_video_path}")
    print(f"Frame interval: {frame_interval} (processing every {frame_interval}th frame)")
    print(f"Buffer time: {buffer_time}s at start and end")
    
    if face_detector_mp is None:
        print("Initializing MediaPipe face detector with high confidence threshold...")
        face_detector_mp = mp_face_detection.FaceDetection(
            model_selection=1,  # Use full-range model
            min_detection_confidence=0.85
        )

    print("\nAnalyzing video properties...")
    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vertical_width = int(frame_height * 9 / 16)
    
    # Calculate frame ranges accounting for buffer
    buffer_frames = int(buffer_time * fps)
    start_frame = buffer_frames
    end_frame = total_frames - buffer_frames
    
    print(f"\nVideo properties:")
    print(f"- Total frames: {total_frames}")
    print(f"- Frames to process: {end_frame - start_frame}")
    print(f"- Buffer frames: {buffer_frames} at start and end")
    print(f"- FPS: {fps}")
    print(f"- Resolution: {frame_width}x{frame_height}")
    print(f"- Target crop width: {vertical_width}")
    
    # Extract audio for voice activity detection from trimmed clip
    print("\nExtracting audio for voice activity detection...")
    # Create temp_files directory if it doesn't exist
    temp_files_dir = os.path.join(os.path.dirname(os.path.dirname(output_video_path)), 'temp_files')
    os.makedirs(temp_files_dir, exist_ok=True)
    
    # Create unique name for temp audio file
    timestamp = int(time.time())
    temp_audio_path = os.path.join(temp_files_dir, f"temp_audio_{timestamp}.wav")
    extractAudio(input_video_path, temp_audio_path)

    print("\nAnalyzing audio for speech detection...")
    audio = AudioSegment.from_wav(temp_audio_path)
    samples = np.array(audio.get_array_of_samples())
    
    chunk_duration_ms = 30
    chunk_size = int(audio.frame_rate * chunk_duration_ms / 1000)
    voice_activities = []
    
    # Only process audio for non-buffer region
    start_sample = int(buffer_time * audio.frame_rate)
    end_sample = len(samples) - int(buffer_time * audio.frame_rate)
    samples = samples[start_sample:end_sample]
    
    total_chunks = len(samples) // chunk_size
    print(f"Processing {total_chunks} audio chunks (excluding buffer regions)...")
    
    for i in range(0, len(samples), chunk_size):
        chunk = samples[i:i + chunk_size]
        if len(chunk) == chunk_size:
            energy = np.sum(chunk ** 2) / len(chunk)
            is_speech = energy > np.mean(samples ** 2) * 1.5
            voice_activities.append(is_speech)

    print("\nStarting face detection analysis...")
    tracker = FaceTracker(frame_width, frame_height, vertical_width)
    frame_count = 0
    face_detected_count = 0
    last_progress = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip detection in buffer regions but still track for smooth transitions
        is_in_buffer = frame_count < buffer_frames or frame_count >= (total_frames - buffer_frames)
        
        # Progress update every 5%
        progress = ((frame_count - buffer_frames) / (end_frame - start_frame)) * 100
        if not is_in_buffer and int(progress) > last_progress and progress % 5 == 0:
            print(f"Processing: {int(progress)}% complete (frame {frame_count - buffer_frames}/{end_frame - start_frame})")
            print(f"Faces detected so far: {face_detected_count}")
            last_progress = int(progress)

        # Only check for speech in non-buffer regions
        voice_activity_index = int((frame_count - buffer_frames) * chunk_duration_ms / (1000 / fps))
        is_speaking = False
        if not is_in_buffer and 0 <= voice_activity_index < len(voice_activities):
            is_speaking = voice_activities[voice_activity_index]

        current_faces = []
        if frame_count % frame_interval == 0 and not is_in_buffer:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector_mp.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    if detection.score[0] > 0.85:
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * frame_width)
                        y = int(bbox.ymin * frame_height)
                        w = int(bbox.width * frame_width)
                        h = int(bbox.height * frame_height)
                        
                        if (w * h) / (frame_width * frame_height) > 0.01:
                            current_faces.append([x, y, w, h])
                            face_detected_count += 1

        crop_x = tracker.update(current_faces, is_speaking)
        if not is_in_buffer:
            Frames.append([int(crop_x), 0, vertical_width, frame_height])
        frame_count += 1

    cap.release()
    
    # Cleanup temp audio file
    if os.path.exists(temp_audio_path):
        try:
            os.remove(temp_audio_path)
            print(f"Cleaned up temporary audio file: {temp_audio_path}")
        except Exception as e:
            print(f"Warning: Could not remove temporary audio file: {e}")

    print("\n=== Face Detection Complete ===")
    print(f"Total frames analyzed: {end_frame - start_frame}")
    print(f"Total faces detected: {face_detected_count}")
    if end_frame - start_frame > 0:
        print(f"Face detection rate: {(face_detected_count/(end_frame - start_frame))*100:.2f}%")
    return Frames

def ensure_valid_crop_window(x_start, x_end, vertical_width, original_width):
    if x_end > original_width:
        x_end = original_width
        x_start = max(0, x_end - vertical_width)
    if x_start < 0:
        x_start = 0
        x_end = min(original_width, vertical_width)
    return x_start, x_end

def detect_face_and_crop(video_path, output_path, start_time, end_time):
    global face_detector_mp
    try:
        print("\n=== Starting Video Processing Pipeline ===")
        print(f"Processing segment from {start_time}s to {end_time}s")
        
        # Clear any existing frames
        Frames.clear()
        
        # Create temporary directory and get temp files path
        base_dir = os.path.dirname(os.path.dirname(output_path))
        temp_files_dir = os.path.join(base_dir, 'temp_files')
        os.makedirs(temp_files_dir, exist_ok=True)
        
        # Create temporary clip for processing with 1 second buffer
        buffer_time = 1.0
        timestamp = int(time.time())
        temp_clip_path = os.path.join(temp_files_dir, f"temp_clip_for_detection_{timestamp}.mp4")
        duration = end_time - start_time
        
        print(f"Adding {buffer_time}s buffer to start and end for processing")
        
        # Create temp clip with buffer
        success, adjusted_start_time = create_temp_clip(
            video_path, 
            temp_clip_path, 
            start_time,
            duration,
            buffer=buffer_time
        )
        
        if not success:
            raise Exception("Failed to create temporary clip")
        
        print("\nBeginning face detection on temporary clip...")
        # Detect faces and speakers on the temporary clip - include buffer_time info
        detect_faces_and_speakers(temp_clip_path, output_path, buffer_time=buffer_time)
        
        # Open temporary video to get properties
        cap = cv2.VideoCapture(temp_clip_path)
        if not cap.isOpened():
            raise Exception("Could not open temporary video file")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vertical_width = int(frame_height * 9 / 16)

        if frame_width < vertical_width:
            raise Exception("Original video too narrow for vertical format")

        # Initialize tracking variables
        x_start = (frame_width - vertical_width) // 2

        print("\nProcessing final video segment (removing buffer)...")
        # Process video segment - start at buffer_time to skip the buffer period
        success = process_video_segment(
            temp_clip_path,
            output_path,
            buffer_time,  # Skip initial buffer
            duration,     # Only process the original duration without buffers
            x_start,
            vertical_width,
            frame_height,
            total_clip_duration=duration + (buffer_time * 2)  # Pass total duration info
        )

        if not success:
            raise Exception("Video processing failed")

        cap.release()
        return True

    except Exception as e:
        print(f"Error in detect_face_and_crop: {str(e)}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        return False

    finally:
        # Cleanup temporary files
        cleanup_resources()
        # Note: We don't delete temp_files directory since it's shared,
        # just clean up our temporary files
        temp_files_dir = os.path.join(os.path.dirname(os.path.dirname(output_path)), 'temp_files')
        if os.path.exists(temp_files_dir):
            try:
                for file in os.listdir(temp_files_dir):
                    if file.startswith(('temp_clip_for_detection_', 'temp_audio_', 'temp_output_')):
                        try:
                            os.remove(os.path.join(temp_files_dir, file))
                        except:
                            pass
            except:
                pass

def cleanup_resources():
    global face_detector_mp
    if face_detector_mp:
        face_detector_mp = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_hardware_encoder():
    try:
        nvidia_output = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if nvidia_output.returncode == 0:
            ffmpeg_output = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True)
            if 'h264_nvenc' in ffmpeg_output.stdout:
                return 'h264_nvenc'
    except:
        pass
    return 'libx264'

def process_video_segment(input_path, output_path, start_time, duration, crop_x, output_width, output_height, total_clip_duration=None):
    try:
        print("\n=== Starting Video Processing ===")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        if total_clip_duration:
            print(f"Total clip duration (with buffer): {total_clip_duration}s")
            print(f"Processing segment: {start_time}s to {start_time + duration}s (removing buffer)")
        print(f"Final output duration: {duration}s")
        print(f"Crop window: {output_width}x{output_height} at x={crop_x}")

        if not os.path.exists(input_path):
            raise Exception(f"Input file does not exist: {input_path}")

        # Use temp_files directory instead of temp_processing
        temp_files_dir = os.path.join(os.path.dirname(os.path.dirname(output_path)), 'temp_files')
        os.makedirs(temp_files_dir, exist_ok=True)
        
        timestamp = int(time.time())
        temp_output = os.path.join(temp_files_dir, f"temp_output_{timestamp}.mp4")
        
        encoder = get_hardware_encoder()
        print(f"\nUsing video encoder: {encoder}")
        print("Starting FFmpeg processing...")

        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_time),  # Start after buffer
            '-i', input_path,
            '-t', str(duration),     # Only take original duration
            '-filter_complex', f'[0:v]crop={output_width}:{output_height}:{crop_x}:0,scale=1080:1920[v]',
            '-map', '[v]',
            '-map', '0:a',
            '-c:v', encoder
        ]

        if encoder == 'h264_nvenc':
            ffmpeg_cmd.extend(['-preset', 'p4', '-rc', 'vbr', '-cq', '23', '-b:v', '5M'])
        else:
            ffmpeg_cmd.extend(['-preset', 'veryfast', '-crf', '23'])

        ffmpeg_cmd.extend([
            '-c:a', 'aac',
            '-b:a', '192k',
            '-ac', '2',
            '-ar', '44100',
            '-movflags', '+faststart',
            temp_output
        ])

        process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            raise Exception(f"FFmpeg process failed: {process.stderr}")

        if os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_output, output_path)
            print(f"\nVideo processing complete: {output_path}")
            return True
        else:
            raise Exception("Output file is missing or empty")

    except Exception as e:
        print(f"Error during video processing: {str(e)}")
        # Clean up temp file if it exists
        if os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except:
                pass
        return False

    finally:
        # Only clean up our specific temp files
        temp_files_dir = os.path.join(os.path.dirname(os.path.dirname(output_path)), 'temp_files')
        if os.path.exists(temp_files_dir):
            try:
                for file in os.listdir(temp_files_dir):
                    if file.startswith('temp_output_'):
                        try:
                            os.remove(os.path.join(temp_files_dir, file))
                        except:
                            pass
            except:
                pass

def create_temp_clip(video_path, temp_path, start_time, duration, buffer=1.0):
    """Creates a temporary clip from the video with buffer time on both ends"""
    try:
        print("\n=== Creating Initial Temp Clip ===")
        start_with_buffer = max(0, start_time - buffer)
        duration_with_buffer = duration + (buffer * 2)
        
        print(f"Original segment: {start_time}s to {start_time + duration}s")
        print(f"With buffer: {start_with_buffer}s to {start_with_buffer + duration_with_buffer}s")
        print(f"Buffer size: {buffer}s on each end")
        
        encoder = get_hardware_encoder()
        print(f"Using encoder: {encoder}")
        print("Starting clip extraction...")
        
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_with_buffer),
            '-i', video_path,
            '-t', str(duration_with_buffer),
            '-c:v', encoder,
            '-c:a', 'aac',
            temp_path
        ]
        
        process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if process.returncode != 0:
            raise Exception(f"FFmpeg process failed: {process.stderr}")
        
        print(f"Temp clip created successfully: {temp_path}")
        print(f"Clip duration: {duration_with_buffer}s")
        
        return True, start_with_buffer
    except Exception as e:
        print(f"Error creating temp clip: {str(e)}")
        return False, 0

# Register cleanup on module exit
import atexit
atexit.register(cleanup_resources)
