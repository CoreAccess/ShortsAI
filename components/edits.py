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

# Check if CUDA is available and user hasn't disabled it
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

# Removed: dlib landmark_predictor initialization since it isn't used

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detector_mp = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

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
        ffmpeg.input(video_path).output(
            audio_path, acodec='libmp3lame', ar=44100, ac=2
        ).run(capture_stdout=True, capture_stderr=True)
        print(f"Audio Extracted To: {audio_path}")
        return audio_path
    except ffmpeg.Error as e:
        print(f"FFMPEG Error: {e.stderr.decode('utf-8')}")
        return None
    except Exception as e:
        print(f"An Error Occurred While Extracting Audio: {e}")
        return None

# ----------------------------------------------------------------------------
# Crop a video file to a 9:16 aspect ratio (portrait) then scale to 1080x1920
# ----------------------------------------------------------------------------


def crop_video(input_file, output_file, start_time, end_time):
    try:
        # Get input video dimensions
        probe = ffmpeg.probe(input_file)
        video_stream = next(
            s for s in probe['streams'] if s['codec_type'] == 'video')
        in_width = int(video_stream['width'])
        in_height = int(video_stream['height'])

        # Calculate the duration of the subclip
        duration = end_time - start_time

        # Calculate dimensions maintaining 9:16 aspect ratio
        # Use the input height as reference and calculate the appropriate width
        output_height = in_height
        output_width = int(output_height * (9/16))

        # Calculate the crop dimensions
        crop_width = output_width
        crop_height = output_height

        # Calculate the x offset to center the crop
        x_offset = (in_width - crop_width) // 2

        # Build the ffmpeg command for cropping and resizing
        ffmpeg_command = (
            ffmpeg
            .input(input_file, ss=start_time, t=duration)
            .filter('crop', w=crop_width, h=crop_height, x=x_offset, y=0)
            .output(output_file, codec='libx264', preset='fast', crf=23)
            .overwrite_output()
        )

        # Run the ffmpeg command
        ffmpeg_command.run(capture_stdout=True, capture_stderr=True)

        print(f"Cropped video saved to: {output_file}")

    except ffmpeg.Error as e:
        print(f"FFMPEG error: {e.stderr.decode('utf-8')}")

    except Exception as e:
        print(f"An error occurred during cropping: {e}")

# ----------------------------------------------------------------------------
# Helper function to get video dimensions
# ----------------------------------------------------------------------------


def get_video_dimensions(video_path):
    probe = ffmpeg.probe(video_path)
    video_stream = next(
        (s for s in probe['streams'] if s['codec_type'] == 'video'), None)
    if video_stream:
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        return width, height
    else:
        return None, None

# ----------------------------------------------------------------------------
# Helper function to calculate the center of a face
# ----------------------------------------------------------------------------


def face_center(face):
    x, y, w, h = face
    return x + w // 2, y + h // 2

# ----------------------------------------------------------------------------
# Helper function to calculate the cropping rectangle
# ----------------------------------------------------------------------------


def calculate_crop_rect(face_x, face_y, frame_width, frame_height, target_aspect_ratio):
    crop_height = frame_height
    crop_width = int(crop_height * target_aspect_ratio)

    x_offset = face_x - crop_width // 2
    x_offset = max(0, min(x_offset, frame_width - crop_width))

    return x_offset, 0, crop_width, crop_height

# ----------------------------------------------------------------------------
# Helper function to generate a smooth transition between two crop rectangles
# ----------------------------------------------------------------------------


def generate_tween_frames(start_rect, end_rect, num_frames):
    x1, y1, w1, h1 = start_rect
    x2, y2, w2, h2 = end_rect

    frames = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        x = int(x1 + (x2 - x1) * t)
        y = int(y1 + (y2 - y1) * t)
        w = int(w1 + (w2 - w1) * t)
        h = int(h1 + (h2 - h1) * t)
        frames.append((x, y, w, h))
    return frames

# ----------------------------------------------------------------------------
# Helper function to interpolate between two positions
# ----------------------------------------------------------------------------


def interpolate_position(pos1, pos2, t):
    """Smooth interpolation between two positions using improved ease-in-out."""
    # Quintic ease-in-out for smoother transitions
    t = t * t * t * (t * (t * 6 - 15) + 10)
    return pos1 + (pos2 - pos1) * t

# ----------------------------------------------------------------------------
# Helper function to interpolate between two positions using cubic easing
# ----------------------------------------------------------------------------


def smooth_interpolate(pos1, pos2, t):
    """Enhanced smoother interpolation using cubic easing."""
    # Improved cubic bezier curve with additional smoothing for 0.25s intervals
    t2 = t * t
    t3 = t2 * t
    # Use a more gradual curve for smoother motion over longer intervals
    t = t3 * (3 - 2 * t)  # Modified hermite interpolation
    return pos1 * (1 - t) + pos2 * t

# ----------------------------------------------------------------------------
# Helper function to analyze audio and determine active speaker
# ----------------------------------------------------------------------------


def analyze_audio_for_speaker(audio_path, frame_count, fps):
    y, sr = librosa.load(audio_path, sr=None)
    frame_duration = 1 / fps
    frame_samples = int(frame_duration * sr)
    energy = np.array([np.sum(np.abs(y[i:i + frame_samples] ** 2)) for i in range(0, len(y), frame_samples)])
    return energy[:frame_count]

# ----------------------------------------------------------------------------
# Detect a face in a video and crop the video around the face
# ----------------------------------------------------------------------------


def extract_audio(input_video, output_audio, start_time, duration):
    """Extract audio in a separate function for better error handling"""
    try:
        if not os.path.exists(input_video):
            print(f"Input video not found: {input_video}")
            return False
            
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_audio), exist_ok=True)
        
        # Extract audio with accurate timing and maintain sync
        command = [
            'ffmpeg', '-y',
            '-i', input_video,
            '-ss', str(start_time),
            '-t', str(duration),
            '-vn',
            '-af', 'asetpts=PTS-STARTPTS',
            '-ac', '2',
            '-ar', '44100',
            '-b:a', '192k',
            output_audio
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error in audio extraction: {result.stderr}")
            return False
            
        # Verify the output file exists and has content
        if not os.path.exists(output_audio) or os.path.getsize(output_audio) == 0:
            print(f"Audio extraction failed: Output file is missing or empty")
            return False
            
        return True
            
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return False

def get_hardware_encoder():
    """Determine the best available hardware encoder"""
    try:
        # Check for NVIDIA GPU
        nvidia_output = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if nvidia_output.returncode == 0:
            # Verify NVENC support
            ffmpeg_output = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True)
            if 'h264_nvenc' in ffmpeg_output.stdout:
                return 'h264_nvenc'
    except:
        pass
    
    # Fallback to CPU encoding
    return 'libx264'

def ensure_directory_exists(path):
    """Ensure directory exists and is writable"""
    try:
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Test write permissions with a temporary file
        test_file = os.path.join(directory, '.write_test')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return True
        except Exception as e:
            print(f"Directory is not writable: {directory}")
            print(f"Error: {str(e)}")
            return False
    except Exception as e:
        print(f"Could not create/verify directory: {directory}")
        print(f"Error: {str(e)}")
        return False

def process_video_segment(input_path, output_path, start_time, duration, crop_x, output_width, output_height):
    """Process a video segment with hardware-accelerated encoding when available"""
    try:
        print("\nStarting optimized video processing...")
        
        # Verify input file exists and is readable
        if not os.path.exists(input_path):
            raise Exception(f"Input file does not exist: {input_path}")
        
        # Ensure output directory exists and is writable
        if not ensure_directory_exists(output_path):
            raise Exception("Cannot write to output directory")
        
        # Create a temporary directory for intermediate files
        temp_dir = os.path.join(os.path.dirname(output_path), 'temp_processing')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Use a unique temporary filename
        temp_output = os.path.join(temp_dir, f"temp_{os.path.basename(output_path)}")
        
        # Get the best available encoder
        encoder = get_hardware_encoder()
        print(f"Using video encoder: {encoder}")
        
        try:
            # Construct FFmpeg command with improved audio handling
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-i', input_path,
                '-t', str(duration),
                '-filter_complex', f'[0:v]crop={output_width}:{output_height}:{crop_x}:0,scale=1080:1920[v]',
                '-map', '[v]',
                '-map', '0:a',
                '-c:v', encoder,
                '-vsync', '1',  # Maintain AV sync
                '-async', '1',  # Audio sync method
            ]
            
            # Add encoder-specific parameters
            if encoder == 'h264_nvenc':
                ffmpeg_cmd.extend([
                    '-preset', 'p4',
                    '-rc', 'vbr',
                    '-cq', '23',
                    '-b:v', '5M'
                ])
            else:
                ffmpeg_cmd.extend([
                    '-preset', 'veryfast',
                    '-crf', '23'
                ])
            
            # Add audio and common parameters with improved sync
            ffmpeg_cmd.extend([
                '-c:a', 'aac',
                '-b:a', '192k',
                '-ac', '2',
                '-ar', '44100',
                '-af', 'aresample=async=1000',  # Handle audio sync issues
                '-movflags', '+faststart',
                '-max_muxing_queue_size', '9999',
                temp_output
            ])
            
            print("Starting FFmpeg process...")
            print(f"Command: {' '.join(ffmpeg_cmd)}")
            
            # Run FFmpeg command
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Monitor encoding progress with timeout
            start_process_time = time.time()
            timeout = 600  # 10 minutes timeout
            last_progress_time = start_process_time
            
            while True:
                # Check if process has finished
                return_code = process.poll()
                if return_code is not None:
                    if return_code != 0:
                        stderr_output = ''.join(process.stderr.readlines())
                        raise Exception(f"FFmpeg process failed with code {return_code}: {stderr_output}")
                    break
                
                # Check for timeout
                current_time = time.time()
                if current_time - start_process_time > timeout:
                    process.kill()
                    raise Exception("Video processing timed out after 10 minutes")
                
                # Read stderr for progress (fixed duplicate code)
                stderr_line = process.stderr.readline()
                if stderr_line:
                    stderr_line = stderr_line.strip()
                    if 'frame=' in stderr_line:
                        print(f"\rEncoding: {stderr_line}", end='', flush=True)
                        last_progress_time = current_time
                    elif 'fps=' in stderr_line or 'speed=' in stderr_line:
                        print(f"\rStats: {stderr_line}", end='', flush=True)
                
                # Check if process is stuck
                if current_time - last_progress_time > 60:  # No progress for 1 minute
                    process.kill()
                    raise Exception("Video processing appears to be stuck - no progress for 1 minute")
                
                time.sleep(0.1)
            
            print("\nVerifying output...")
            # If successful, move temp file to final location
            if os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
                # Verify the temp output is a valid video
                try:
                    probe = ffmpeg.probe(temp_output)
                    if any(s['codec_type'] == 'video' for s in probe['streams']):
                        print("Output validation successful")
                        # Move temp file to final location
                        if os.path.exists(output_path):
                            os.remove(output_path)
                        os.rename(temp_output, output_path)
                        print("\nVideo processing complete!")
                        return True
                    else:
                        raise Exception("Temporary output file is not a valid video")
                except ffmpeg.Error as e:
                    raise Exception(f"Invalid temporary output file: {str(e)}")
            else:
                raise Exception("Temporary output file is missing or empty")
            
        finally:
            # Clean up temporary directory
            try:
                if os.path.exists(temp_dir):
                    for file in os.listdir(temp_dir):
                        try:
                            os.remove(os.path.join(temp_dir, file))
                        except Exception as e:
                            print(f"Warning: Could not remove temp file {file}: {e}")
                    os.rmdir(temp_dir)
            except Exception as e:
                print(f"Warning: Could not clean up temp directory: {e}")
        
    except Exception as e:
        print(f"\nError during video processing: {str(e)}")
        if isinstance(e, ffmpeg.Error):
            print(f"FFmpeg error details: {e.stderr.decode('utf-8') if e.stderr else 'No stderr output'}")
        return False

def calculate_face_score(face, audio_energy, frame_idx, frame_width, frame_height, face_history=None):
    """Calculate a score for each detected face with improved weighting for active speakers"""
    x, y, w, h = face
    face_size = w * h
    face_center_x = x + w/2
    face_center_y = y + h/2
    
    # Size score (0-1) - prefer larger faces but with diminishing returns
    size_ratio = face_size / (frame_width * frame_height)
    size_score = min(1.0, size_ratio * 3)  # Boost small-medium faces
    
    # Framing score (0-1) - penalize partial faces and shoulders
    border_margin = 0.1 * frame_width
    fully_visible = (x > border_margin and 
                    x + w < frame_width - border_margin and 
                    y > 0 and 
                    y + h < frame_height)
    framing_score = 1.0 if fully_visible else 0.3 
    
    # Center weighting (0-1) - prefer faces closer to vertical center
    center_dist_x = abs(face_center_x - frame_width/2) / frame_width
    center_dist_y = abs(face_center_y - frame_height/2) / frame_height
    center_score = 1 - (center_dist_x * 0.7 + center_dist_y * 0.3)
    
    # Audio activity score (0-1) with temporal smoothing
    audio_window = 5  # Look at surrounding frames
    start_idx = max(0, frame_idx - audio_window)
    end_idx = min(len(audio_energy), frame_idx + audio_window + 1)
    if start_idx < end_idx and len(audio_energy) > 0:
        recent_energy = audio_energy[start_idx:end_idx]
        audio_score = min(1.0, np.mean(recent_energy) / np.max(audio_energy))
    else:
        audio_score = 0.5

    # Consistency score (0-1) - reward faces that maintain position
    consistency_score = 0.5
    if face_history and len(face_history) > 0:
        prev_centers = [(f[0] + f[2]/2, f[1] + f[3]/2) for f in face_history[-5:]]
        if prev_centers:
            distances = [np.sqrt((cx - face_center_x)**2 + (cy - face_center_y)**2) 
                        for cx, cy in prev_centers]
            avg_movement = np.mean(distances) if distances else frame_width
            consistency_score = 1.0 - min(1.0, avg_movement / (frame_width * 0.5))

    # Weighted combination with emphasis on framing and audio activity
    weights = {
        'size': 0.15,
        'framing': 0.25,
        'center': 0.15,
        'audio': 0.30,
        'consistency': 0.15
    }
    
    final_score = (
        weights['size'] * size_score +
        weights['framing'] * framing_score +
        weights['center'] * center_score +
        weights['audio'] * audio_score +
        weights['consistency'] * consistency_score
    )
    
    return final_score

def detect_scene_change(prev_frame, curr_frame, threshold=30.0):
    """Detect if there's a scene change between frames using optimized comparison"""
    if prev_frame is None or curr_frame is None:
        return False
    
    try:
        # Ensure frames are valid
        if prev_frame.size == 0 or curr_frame.size == 0:
            return False
        
        # Downscale frames for faster comparison
        scale_factor = 0.25
        small_prev = cv2.resize(prev_frame, None, fx=scale_factor, fy=scale_factor)
        small_curr = cv2.resize(curr_frame, None, fx=scale_factor, fy=scale_factor)
            
        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(small_prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(small_curr, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(prev_gray, curr_gray)
        mean_diff = np.mean(frame_diff)
        
        return mean_diff > threshold
    except Exception as e:
        print(f"Warning: Scene change detection failed: {str(e)}")
        return False

def process_frame_batch(frames, frame_indices, audio_energy, frame_width, frame_height, face_history):
    """Process a batch of frames in parallel with optimized memory usage"""
    results = []
    
    # Process frames in smaller chunks to manage memory better
    chunk_size = 4
    for i in range(0, len(frames), chunk_size):
        chunk_frames = frames[i:i + chunk_size]
        chunk_indices = frame_indices[i:i + chunk_size]
        
        chunk_results = []
        for frame, frame_idx in zip(chunk_frames, chunk_indices):
            if frame is None or frame.size == 0:
                continue
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = face_detector_mp.process(frame_rgb)
            
            # Clear memory
            del frame_rgb
            
            current_faces = []
            if detections.detections:
                for detection in detections.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * frame_width)
                    y = int(bbox.ymin * frame_height)
                    w = int(bbox.width * frame_width)
                    h = int(bbox.height * frame_height)
                    
                    face = (x, y, w, h)
                    score = calculate_face_score(
                        face,
                        audio_energy,
                        frame_idx,
                        frame_width,
                        frame_height,
                        face_history
                    )
                    current_faces.append((face, score))
            
            chunk_results.append((frame_idx, current_faces))
            
        results.extend(chunk_results)
        
        # Force garbage collection after each chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results

def cleanup_resources():
    """Cleanup MediaPipe and other resources"""
    global face_detector_mp
    if face_detector_mp:
        face_detector_mp.close()
        face_detector_mp = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def detect_face_and_crop(video_path, output_path, start_time, end_time):
    try:
        # Verify input file exists and is readable
        if not os.path.exists(video_path):
            raise Exception(f"Input video file does not exist: {video_path}")
            
        # Ensure output directory exists and is writable
        if not ensure_directory_exists(output_path):
            raise Exception("Cannot write to output directory")
        
        # Validate duration and adjust times
        probe = ffmpeg.probe(video_path) 
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        total_duration = float(probe['format']['duration'])
        start_time = max(0, min(float(start_time), total_duration))
        end_time = max(start_time, min(float(end_time), total_duration))
        duration = end_time - start_time

        # Calculate dimensions
        frame_width = int(video_info['width'])
        frame_height = int(video_info['height'])
        output_height = frame_height
        output_width = int(output_height * (9/16))

        # Extract audio for speaker detection with better error handling
        temp_audio = os.path.join(os.path.dirname(output_path), "temp_audio.wav")
        audio_energy = []
        
        audio_extracted = extract_audio(video_path, temp_audio, start_time, duration)
        if audio_extracted:
            try:
                # Get audio energy profile with error handling
                y, sr = librosa.load(temp_audio, sr=None)
                if y is not None and len(y) > 0:
                    hop_length = int(sr / 30)  # For 30fps video
                    audio_energy = librosa.feature.rms(y=y, hop_length=hop_length)[0]
                else:
                    print("Warning: Audio loaded but empty, continuing without audio analysis")
            except Exception as e:
                print(f"Warning: Audio analysis failed: {str(e)}, continuing without audio analysis")
        else:
            print("Warning: Audio extraction failed, continuing without audio analysis")

        # Clean up audio file if it exists
        if os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except Exception as e:
                print(f"Warning: Failed to clean up temp audio file: {str(e)}")

        # Initialize face tracking with enhanced history
        face_history = []  # Store full face information
        position_history = []  # Store crop positions
        smoothing_window = 15  # Increased window for smoother transitions
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        frame_idx = 0
        
        # Track scene changes
        scene_change_frames = []
        last_scene_change = 0
        
        # Buffer for face scoring
        face_score_buffer = []
        score_buffer_size = 5
        
        # Initialize frame processing
        frame_buffer_size = 8  # Process frames in batches of 8
        frame_buffer = deque(maxlen=frame_buffer_size)
        frame_idx_buffer = deque(maxlen=frame_buffer_size)
        
        # Get total frame count for progress tracking
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0
        last_progress_update = 0
        process_start_time = time.time()  # Renamed to avoid conflict with start_time parameter
        
        print(f"\nStarting frame analysis of {total_frames:,} frames...")
        print("Progress: [" + "-" * 20 + "]")
        print("          ", end="", flush=True)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_batch = {}
            
            # Read first frame to initialize
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                raise Exception("Could not read valid first frame")
            
            prev_frame = frame.copy()
            processed_frames += 1
            
            # Initialize buffers with first frame if it's valid
            if frame is not None and frame.size > 0:
                frame_buffer.append(frame.copy())
                frame_idx_buffer.append(frame_idx)
                frame_idx += 1
            
            # Scene detection optimization
            scene_check_interval = 5  # Check every 5 frames for scene changes
            scene_check_counter = 0
            
            while cap.get(cv2.CAP_PROP_POS_MSEC) <= end_time * 1000:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    print("\nReached end of video or invalid frame")
                    break
                
                processed_frames += 1
                
                # Update progress bar every 5%
                progress = (processed_frames / total_frames) * 100
                if progress - last_progress_update >= 5:
                    print("█", end="", flush=True)
                    last_progress_update = progress
                
                # Print detailed status every 30 frames
                if processed_frames % 30 == 0:
                    elapsed_time = time.time() - process_start_time
                    frames_per_second = processed_frames / elapsed_time
                    remaining_frames = total_frames - processed_frames
                    estimated_remaining = remaining_frames / frames_per_second if frames_per_second > 0 else 0
                    
                    print(f"\rFrame: {processed_frames:,}/{total_frames:,} ({progress:,.1f}%) | "
                          f"ETA: {estimated_remaining:.1f}s | "
                          f"Speed: {frames_per_second:.1f} fps | "
                          f"Batches: {len(future_to_batch)}", end="    \r", flush=True)
                
                # Buffer frames for batch processing
                frame_buffer.append(frame.copy())
                frame_idx_buffer.append(frame_idx)
                
                # Only check for scene changes periodically to improve performance
                if scene_check_counter % scene_check_interval == 0:
                    is_scene_change = detect_scene_change(prev_frame, frame)
                    if is_scene_change:
                        scene_change_frames.append(frame_idx)
                        face_history = []
                        position_history = []
                        face_score_buffer = []
                        last_scene_change = frame_idx
                    prev_frame = frame.copy()
                scene_check_counter += 1
                
                # Process batch when buffer is full
                if len(frame_buffer) == frame_buffer_size:
                    future = executor.submit(
                        process_frame_batch,
                        list(frame_buffer),
                        list(frame_idx_buffer),
                        audio_energy,
                        frame_width,
                        frame_height,
                        face_history
                    )
                    future_to_batch[future] = list(frame_idx_buffer)
                    print(f"\rProcessing batch {len(future_to_batch)} | Frame {processed_frames}/{total_frames} ({progress:.1f}%)", end="", flush=True)
                    frame_buffer.clear()
                    frame_idx_buffer.clear()
                
                # Process completed batches
                completed_batches = 0
                for future in list(future_to_batch.keys()):
                    if future.done():
                        batch_results = future.result()
                        completed_batches += 1
                        for frame_idx, current_faces in batch_results:
                            if current_faces:
                                # Sort faces by score
                                current_faces.sort(key=lambda x: x[1], reverse=True)
                                best_face = current_faces[0][0]
                                
                                face_history.append(best_face)
                                if len(face_history) > smoothing_window:
                                    face_history.pop(0)
                                
                                # Calculate crop position with scene awareness
                                face_center_x = best_face[0] + best_face[2] // 2
                                
                                if position_history:
                                    frames_since_scene_change = frame_idx - last_scene_change
                                    max_movement = (
                                        frame_width * 0.5 if frames_since_scene_change < 5
                                        else frame_width * 0.02
                                    )
                                    
                                    prev_x = position_history[-1]
                                    delta = face_center_x - prev_x
                                    
                                    if abs(delta) > max_movement:
                                        movement_factor = min(1.0, max_movement / abs(delta))
                                        face_center_x = prev_x + (delta * movement_factor)
                                
                                position_history.append(face_center_x)
                                if len(position_history) > smoothing_window:
                                    position_history.pop(0)
                        
                        del future_to_batch[future]
                
                frame_idx += 1
            
            print("]")
            print("\nProcessing remaining frames...")
            
            # Process remaining frames in buffer
            if frame_buffer:
                batch_results = process_frame_batch(
                    list(frame_buffer),
                    list(frame_idx_buffer),
                    audio_energy,
                    frame_width,
                    frame_height,
                    face_history
                )
                # Process final batch results similarly to above
            
            print("\nFrame analysis complete!")
            print("Finalizing video processing...")
            
            cap.release()

        # Clean up temporary audio file
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

        # Calculate final crop position with improved smoothing
        if position_history:
            # Use exponential moving average for final positions
            alpha = 0.3  # Smoothing factor
            smoothed_positions = []
            current_smooth = position_history[0]
            
            for pos in position_history:
                current_smooth = alpha * pos + (1 - alpha) * current_smooth
                smoothed_positions.append(current_smooth)
            
            center_x = int(smoothed_positions[-1])
            crop_x = max(0, min(center_x - output_width // 2, frame_width - output_width))
        else:
            crop_x = (frame_width - output_width) // 2

        print("\nStarting optimized video encoding...")
        # Process video with better error handling
        success = process_video_segment(
            video_path,
            output_path,
            start_time,
            duration,
            crop_x,
            output_width,
            output_height
        )
        
        if not success:
            raise Exception("Video processing failed - check previous error messages")
            
        # Final validation of output
        if not os.path.exists(output_path):
            raise Exception("Output file was not created during processing")
            
        if os.path.getsize(output_path) == 0:
            raise Exception("Output file is empty after processing")
            
        try:
            # Verify the output is a valid video file
            probe = ffmpeg.probe(output_path)
            if not any(s['codec_type'] == 'video' for s in probe['streams']):
                raise Exception("Output file does not contain a valid video stream")
        except ffmpeg.Error as e:
            raise Exception(f"Invalid output video file: {str(e)}")
        
        print(f"\nSuccessfully created video at: {output_path}")
        return True

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # Clean up any partial output
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                print("Cleaned up partial output file")
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up partial output: {cleanup_error}")
        raise
    finally:
        cleanup_resources()

# Cleanup on module exit
import atexit
atexit.register(cleanup_resources)
