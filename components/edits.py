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
global Frames
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
        
        # Ensure the temp directory exists
        os.makedirs(temp_files_dir, exist_ok=True)
        
        audio = AudioSegment.from_file(video_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(audio_path, format="wav")
        print(f"Audio extracted successfully to: {audio_path}")
        
        # Register the audio file for cleanup later
        register_temp_file(audio_path)
        
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

def create_temp_clip(video_path, temp_path, start_time, duration, buffer=1.0):
    """Creates a temporary clip from the video with buffer time on both ends"""
    try:
        print("\n=== Creating Initial Temp Clip ===")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")
            
        start_with_buffer = max(0, start_time - buffer)
        duration_with_buffer = duration + (buffer * 2)
        
        print(f"Original segment: {start_time}s to {start_time + duration}s")
        print(f"With buffer: {start_with_buffer}s to {start_with_buffer + duration_with_buffer}s")
        print(f"Buffer size: {buffer}s on each end")
        
        encoder = get_hardware_encoder()
        print(f"Using encoder: {encoder}")
        print("Starting clip extraction...")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
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
            print("FFmpeg stderr output:")
            print(process.stderr)
            raise Exception(f"FFmpeg process failed")
        
        if not os.path.exists(temp_path):
            raise FileNotFoundError(f"FFmpeg did not create the temporary clip at: {temp_path}")
            
        if os.path.getsize(temp_path) == 0:
            raise Exception(f"FFmpeg created an empty temporary clip")
        
        print(f"Temp clip created successfully: {temp_path}")
        print(f"Clip duration: {duration_with_buffer}s")
        print(f"File size: {os.path.getsize(temp_path) / (1024*1024):.2f} MB")
        
        return True, start_with_buffer
        
    except Exception as e:
        print(f"\nError creating temp clip: {str(e)}")
        return False, 0

class FaceTracker:
    def __init__(self, frame_width, frame_height, vertical_width):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.vertical_width = vertical_width
        self.history_size = 15  # 0.5 seconds at 30fps
        self.position_history = deque(maxlen=self.history_size)
        self.face_history = deque(maxlen=self.history_size)
        self.no_face_counter = 0
        self.max_no_face_frames = 30  # 1 second at 30fps
        self.last_valid_position = None
        self.center_position = (frame_width - vertical_width) // 2
        self.confidence_threshold = 0.85
        self.movement_smoothing = 0.4
        self.speaker_bonus = 1.5
        self.min_movement_threshold = 10
        self.max_movement_per_frame = vertical_width * 0.3

    def update(self, faces, is_speaking=False):
        if not faces:
            self.no_face_counter += 1
            if self.no_face_counter >= self.max_no_face_frames:
                # Transition to center when no face detected
                alpha = min(1.0, (self.no_face_counter - self.max_no_face_frames) / 15)
                if self.last_valid_position is None:
                    return self.center_position
                position = int(self.last_valid_position * (1 - alpha) + self.center_position * alpha)
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
            
            # Enhanced scoring system
            size_score = (w * h) / (self.frame_width * self.frame_height)
            center_offset = abs(center_x - self.frame_width/2)
            center_score = 1 - (center_offset / (self.frame_width/2))
            vertical_score = 1 - (y / self.frame_height)
            
            score = (
                size_score * 0.5 +
                center_score * 0.4 +
                vertical_score * 0.1
            )
            
            if is_speaking:
                score *= 2.0
            
            if score > best_score:
                best_score = score
                best_face = face

        if best_score < 0.2:
            if self.last_valid_position is not None:
                return self.last_valid_position
            return self.center_position

        x, y, w, h = best_face
        face_center = x + w/2

        # Calculate ideal crop position
        ideal_crop_x = max(0, min(self.frame_width - self.vertical_width,
                                 face_center - self.vertical_width/2))
        
        # Apply margin based on face size
        margin = w * 0.2
        if x - margin < ideal_crop_x:
            ideal_crop_x = max(0, x - margin)
        elif (x + w + margin) > (ideal_crop_x + self.vertical_width):
            ideal_crop_x = min(self.frame_width - self.vertical_width,
                             x + w + margin - self.vertical_width)
        
        # If we have a previous position, limit maximum movement per frame
        if self.last_valid_position is not None:
            movement = ideal_crop_x - self.last_valid_position
            if abs(movement) < self.min_movement_threshold:
                return self.last_valid_position
                
            movement = max(-self.max_movement_per_frame, 
                         min(self.max_movement_per_frame, movement))
            ideal_crop_x = self.last_valid_position + movement
        
        # Ensure ideal_crop_x is a valid number
        if not isinstance(ideal_crop_x, (int, float)) or np.isnan(ideal_crop_x):
            return self.center_position if self.last_valid_position is None else self.last_valid_position
        
        # Shorter history for weighted average
        self.position_history.append(ideal_crop_x)
        if len(self.position_history) >= 3:
            weights = np.exp(np.linspace(-0.5, 0, len(self.position_history)))
            weights /= weights.sum()
            try:
                smoothed_position = int(np.average(self.position_history, weights=weights))
            except:
                smoothed_position = int(ideal_crop_x)
        else:
            smoothed_position = int(ideal_crop_x)
        
        # Additional boundary check
        smoothed_position = max(0, min(self.frame_width - self.vertical_width, smoothed_position))
        
        self.last_valid_position = smoothed_position
        return smoothed_position

def detect_faces_and_speakers(input_video_path, output_video_path, frame_interval=1, buffer_time=1.0, debug_visualization=True):
    """ 
    Enhanced face detection with better tracking and speaker detection.
    Works with pre-trimmed video clip, accounting for buffer time.
    debug_visualization: If True, shows real-time detection visualization and saves debug video
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
    if not cap.isOpened():
        raise Exception("Failed to open video file")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vertical_width = int(frame_height * 9 / 16)
    
    if fps <= 0 or frame_width <= 0 or frame_height <= 0:
        raise Exception("Invalid video properties detected")
    
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
    temp_files_dir = os.path.join(os.path.dirname(os.path.dirname(output_video_path)), 'temp_files')
    os.makedirs(temp_files_dir, exist_ok=True)
    
    timestamp = int(time.time())
    temp_audio_path = os.path.join(temp_files_dir, f"temp_audio_{timestamp}.wav")
    if not extractAudio(input_video_path, temp_audio_path):
        raise Exception("Failed to extract audio from video")

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
    
    # Compute mean energy once for the entire sample
    mean_energy = np.mean(samples ** 2)
    
    for i in range(0, len(samples), chunk_size):
        chunk = samples[i:i + chunk_size]
        if len(chunk) == chunk_size:
            energy = np.sum(chunk ** 2) / len(chunk)
            is_speech = energy > mean_energy * 1.5
            voice_activities.append(is_speech)

    print("\nStarting face detection analysis...")
    tracker = FaceTracker(frame_width, frame_height, vertical_width)
    frame_count = 0
    face_detected_count = 0
    last_progress = 0
    
    # Setup debug video writer if enabled
    debug_writer = None
    if debug_visualization:
        cv2.namedWindow('Face and Speaker Detection Debug', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Face and Speaker Detection Debug', 960, 540)
        
        debug_output_path = os.path.join(
            os.path.dirname(os.path.dirname(output_video_path)),
            'finished_videos',
            f"debug_{os.path.basename(output_video_path)}"
        )
        print(f"\nSaving debug visualization to: {debug_output_path}")
        
        debug_writer = cv2.VideoWriter(
            debug_output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )
    
    # Default frame data for when no faces are detected
    default_frame_data = [frame_width // 2 - vertical_width // 2, 0, vertical_width, frame_height]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        is_in_buffer = frame_count < buffer_frames or frame_count >= (total_frames - buffer_frames)
        
        # Progress update every 5%
        progress = ((frame_count - buffer_frames) / (end_frame - start_frame)) * 100
        if not is_in_buffer and int(progress) > last_progress and progress % 5 == 0:
            print(f"Processing: {int(progress)}% complete (frame {frame_count - buffer_frames}/{end_frame - start_frame})")
            print(f"Faces detected so far: {face_detected_count}")
            last_progress = int(progress)

        voice_activity_index = int((frame_count - buffer_frames) * chunk_duration_ms / (1000 / fps))
        is_speaking = False
        if not is_in_buffer and 0 <= voice_activity_index < len(voice_activities):
            is_speaking = voice_activities[voice_activity_index]

        current_faces = []
        debug_frame = frame.copy() if debug_visualization else None
        
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
                            
                            if debug_visualization:
                                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), 
                                           (0, 255, 0) if is_speaking else (0, 165, 255), 2)
                                
                                status_text = f"Speaking: {is_speaking}"
                                conf_text = f"Conf: {detection.score[0]:.2f}"
                                
                                cv2.putText(debug_frame, status_text, (x, y - 25),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                          (0, 255, 0) if is_speaking else (0, 165, 255), 2)
                                cv2.putText(debug_frame, conf_text, (x, y - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                          (0, 255, 0) if is_speaking else (0, 165, 255), 2)

        crop_x = tracker.update(current_faces, is_speaking)
        if crop_x is None:  # Handle case where tracker returns None
            frame_data = default_frame_data
        else:
            frame_data = [int(crop_x), 0, vertical_width, frame_height]
            
        if not is_in_buffer:
            Frames.append(frame_data)
            
            if debug_visualization:
                cv2.rectangle(debug_frame, 
                            (frame_data[0], 0), 
                            (frame_data[0] + frame_data[2], frame_height),
                            (255, 0, 0), 2)
                            
        if debug_visualization:
            cv2.imshow('Face and Speaker Detection Debug', debug_frame)
            debug_writer.write(debug_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
        frame_count += 1

    cap.release()
    if debug_visualization:
        if debug_writer:
            debug_writer.release()
        cv2.destroyAllWindows()
        print(f"\nDebug visualization saved to: {debug_output_path}")
    
    # Cleanup temp audio file
    if os.path.exists(temp_audio_path):
        try:
            os.remove(temp_audio_path)
            print(f"Cleaned up temporary audio file: {temp_audio_path}")
        except Exception as e:
            print(f"Warning: Could not remove temporary audio file: {e}")

    if not Frames:  # If no frames were processed, use default center crop
        print("\nWarning: No face tracking data generated, using center crop")
        Frames = [default_frame_data] * (end_frame - start_frame)

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

def detect_face_and_crop(video_path, output_path, start_time, end_time, debug_visualization=True):
    global face_detector_mp, Frames
    try:
        print("\n=== Starting Video Processing Pipeline ===")
        print(f"Processing segment from {start_time}s to {end_time}s")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")
        
        # Initialize frames list
        Frames = []
        
        # Get video properties first
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Create temporary directory and get temp files path
        base_dir = os.path.dirname(os.path.dirname(output_path))
        temp_files_dir = os.path.join(base_dir, 'temp_files')
        os.makedirs(temp_files_dir, exist_ok=True)
        
        # Register temporary directory with cleanup system
        if 'register_temp_file' in globals():
            register_temp_file(temp_files_dir)
        
        # Create temporary clip for processing with 1 second buffer
        buffer_time = 1.0
        timestamp = int(time.time())
        temp_clip_path = os.path.join(temp_files_dir, f"temp_clip_for_detection_{timestamp}.mp4")
        
        # Register temp clip for cleanup
        if 'register_temp_file' in globals():
            register_temp_file(temp_clip_path)
        
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
            
        if not os.path.exists(temp_clip_path):
            raise FileNotFoundError(f"Temporary clip not found after creation: {temp_clip_path}")
        
        print("\nBeginning face detection on temporary clip...")
        detect_faces_and_speakers(temp_clip_path, output_path, buffer_time=buffer_time, debug_visualization=debug_visualization)
        
        # Validate that we have enough frame data
        if not Frames:
            raise Exception("No face tracking data generated during detection")
        
        print(f"Face detection complete. Generated {len(Frames)} frame positions.")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Process final video segment
        print("\nProcessing final video segment...")
        success = process_video_segment(
            temp_clip_path,
            output_path,
            buffer_time,
            duration,
            Frames[0][0] if Frames else 0,  # Use first frame's x position or 0
            Frames[0][2] if Frames else int(frame_height * 9/16),  # Use first frame's width or calculate
            Frames[0][3] if Frames else frame_height,  # Use first frame's height or original
            total_clip_duration=duration + (buffer_time * 2)
        )
        
        if not success:
            raise Exception("Failed to process final video segment")
            
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Final processed video not found: {output_path}")
        
        return True

    except Exception as e:
        print(f"\nError in detect_face_and_crop: {str(e)}")
        print("\nFile status:")
        print(f"Input video exists: {os.path.exists(video_path)}")
        if 'temp_clip_path' in locals():
            print(f"Temp clip exists: {os.path.exists(temp_clip_path)}")
        print(f"Output path exists: {os.path.exists(output_path)}")
        print(f"Number of tracked frames: {len(Frames)}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        return False

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

def interpolate_positions(frame_positions, max_movement=30, smoothing_window=7):
    """Enhanced smoothing of frame-to-frame movements using Gaussian smoothing"""
    if not frame_positions:
        return []
        
    smoothed = []
    positions = np.array([pos[0] for pos in frame_positions])
    
    # Create Gaussian kernel for smoothing
    kernel = np.exp(-0.5 * np.square(np.linspace(-2, 2, smoothing_window)))
    kernel = kernel / np.sum(kernel)
    
    # Pad the positions array for smoothing at edges
    pad_width = smoothing_window // 2
    padded_positions = np.pad(positions, (pad_width, pad_width), mode='edge')
    
    # Apply Gaussian smoothing
    smoothed_x = np.zeros_like(positions)
    for i in range(len(positions)):
        window = padded_positions[i:i + smoothing_window]
        smoothed_x[i] = np.sum(window * kernel)
    
    # Limit frame-to-frame movement
    limited_x = np.zeros_like(smoothed_x)
    limited_x[0] = smoothed_x[0]
    for i in range(1, len(smoothed_x)):
        movement = smoothed_x[i] - limited_x[i-1]
        if abs(movement) > max_movement:
            limited_x[i] = limited_x[i-1] + np.sign(movement) * max_movement
        else:
            limited_x[i] = smoothed_x[i]
    
    # Reconstruct frame data with smoothed x positions
    for i, (x, original) in enumerate(zip(limited_x, frame_positions)):
        smoothed.append([int(x), original[1], original[2], original[3]])
    
    return smoothed

def preview_tracking(frame, x, width, height, is_speaking=False, return_frame=False):
    """
    Helper function to show tracking preview
    return_frame: If True, returns the preview frame instead of showing it
    """
    preview = frame.copy()
    # Draw crop region
    cv2.rectangle(preview, 
                 (int(x), 0), 
                 (int(x + width), height),
                 (255, 0, 0), 2)
    
    # Add tracking info
    cv2.putText(preview, f"Crop X: {int(x)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(preview, f"Width: {width}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    if return_frame:
        return preview
        
    # Show preview
    cv2.imshow('Processing Preview', preview)
    cv2.waitKey(1)

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
        if os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except:
                pass
        return False

    finally:
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

def process_chunk(input_path, output_path, start_time, duration, frames, frame_width, frame_height, fps, encoder):
    """Process a smaller chunk of the video to manage memory usage"""
    try:
        filter_complex = []
        for i, (x, y, w, h) in enumerate(frames):
            # Validate and clamp values
            x = max(0, min(frame_width - w, x))
            y = max(0, min(frame_height - h, y))
            w = min(frame_width - x, w)
            h = min(frame_height - y, h)
            
            pts = f"PTS-STARTPTS" if i == 0 else f"PTS-STARTPTS+{i/fps:.3f}/TB"
            filter_complex.append(f"[0:v]trim=start_frame={int(start_time * fps) + i}:end_frame={int(start_time * fps) + i + 1},setpts={pts},crop={w}:{h}:{x}:{y}[v{i}]")
        
        # Add concatenation filter
        v_labels = ''.join(f"[v{i}]" for i in range(len(frames)))
        filter_complex.append(f"{v_labels}concat=n={len(frames)}:v=1[outv]")
        
        # Join filters with semicolons
        filter_str = ';'.join(filter for filter in filter_complex if filter.strip())
        
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-filter_complex', filter_str,
            '-map', '[outv]',
            '-map', '0:a',
            '-ss', str(start_time),
            '-t', str(duration),
            '-c:v', encoder
        ]

        if encoder == 'h264_nvenc':
            ffmpeg_cmd.extend([
                '-preset', 'p4',
                '-rc', 'vbr',
                '-cq', '23',
                '-b:v', '5M',
                '-maxrate', '10M',
                '-bufsize', '10M'
            ])
        else:
            ffmpeg_cmd.extend([
                '-preset', 'veryfast',
                '-crf', '23'
            ])

        ffmpeg_cmd.extend([
            '-c:a', 'aac',
            '-b:a', '192k',
            '-ac', '2',
            '-ar', '44100',
            '-movflags', '+faststart',
            output_path
        ])

        process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=120)
        if process.returncode != 0:
            print(f"\nFFmpeg Chunk Error Output:")
            print(process.stderr)
            return False
            
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1024:
            return False
            
        return True
        
    except Exception as e:
        print(f"Error processing chunk: {str(e)}")
        return False

def voice_activity_detection(audio_frame, sample_rate=16000):
    """Detect voice activity in an audio frame"""
    try:
        # Simple energy-based voice activity detection
        frame = np.frombuffer(audio_frame, dtype=np.int16)
        energy = np.sum(frame.astype(np.float32) ** 2) / len(frame)
        threshold = 100000  # Adjust this threshold based on your needs
        return energy > threshold
    except Exception as e:
        print(f"Error in voice activity detection: {e}")
        return False

def analyze_audio_for_speech(audio_path, chunk_duration_ms=30):
    """Analyze audio file for speech segments"""
    try:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        samples = np.array(audio.get_array_of_samples())
        
        chunk_size = int(audio.frame_rate * chunk_duration_ms / 1000)
        voice_activities = []
        
        # Calculate average energy for normalization
        mean_energy = np.mean(samples ** 2)
        
        for i in range(0, len(samples), chunk_size):
            chunk = samples[i:i + chunk_size]
            if len(chunk) == chunk_size:
                energy = np.sum(chunk ** 2) / len(chunk)
                is_speech = energy > mean_energy * 1.5
                voice_activities.append(is_speech)
                
        return voice_activities
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return []

def preview_and_save_debug_frame(frame, faces, is_speaking, crop_x, vertical_width, frame_height, debug_writer=None):
    """Create and optionally save a debug visualization frame"""
    debug_frame = frame.copy()
    
    # Draw faces
    for face in faces:
        x, y, w, h = face
        color = (0, 255, 0) if is_speaking else (0, 165, 255)
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, 2)
        
        # Add speaking status
        status_text = f"Speaking: {is_speaking}"
        cv2.putText(debug_frame, status_text, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw crop region
    cv2.rectangle(debug_frame, 
                 (int(crop_x), 0), 
                 (int(crop_x + vertical_width), frame_height),
                 (255, 0, 0), 2)
    
    # Add crop position info
    cv2.putText(debug_frame, f"Crop X: {int(crop_x)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save frame if writer is provided
    if debug_writer is not None:
        debug_writer.write(debug_frame)
    
    return debug_frame

def create_debug_video_writer(output_path, frame_width, frame_height, fps):
    """Create a video writer for debug visualization"""
    debug_output_path = os.path.join(
        os.path.dirname(os.path.dirname(output_path)),
        'finished_videos',
        f"debug_{os.path.basename(output_path)}"
    )
    
    writer = cv2.VideoWriter(
        debug_output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )
    
    return writer, debug_output_path

def register_temp_file(filepath):
    """
    Register a file for manual cleanup. This no longer triggers automatic cleanup,
    instead it just tracks files that need to be cleaned up when processing is complete.
    """
    global _temp_files_created
    if '_temp_files_created' not in globals():
        global _temp_files_created
        _temp_files_created = set()
    _temp_files_created.add(filepath)
    return filepath

# Global set to track temporary files
_temp_files_created = set()
