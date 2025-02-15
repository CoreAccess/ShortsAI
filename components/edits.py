import cv2
import ffmpeg
import os
import tempfile
import torch
import gc
from facenet_pytorch import MTCNN
import numpy as np

# Check if CUDA is available and user hasn't disabled it
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
print(f"MTCNN Using device: {device}")

# Initialize MTCNN with explicit device placement and batching
mtcnn = MTCNN(
    keep_all=True,
    device=device,
    select_largest=True,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    post_process=True,
    image_size=160,
    margin=0,  # No margin to avoid unnecessary padding
    selection_method='probability'  # Use probability-based selection
).to(device)  # Ensure model is on the correct device

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

        # Define the output dimensions (portrait: 1080x1920)
        output_width, output_height = 1080, 1920
        target_aspect_ratio = 9 / 16

        # Calculate the crop dimensions
        crop_width = int(in_height * target_aspect_ratio)
        crop_height = in_height

        # Calculate the x offset to center the crop
        x_offset = (in_width - crop_width) // 2

        # Build the ffmpeg command for cropping and resizing
        ffmpeg_command = (
            ffmpeg
            .input(input_file, ss=start_time, t=duration)
            .filter('crop', w=crop_width, h=crop_height, x=x_offset, y=0)
            .filter('scale', width=output_width, height=output_height)
            .output(output_file, codec='libx264', preset='fast', crf=23)
            .overwrite_output()
        )

        # Run the ffmpeg command
        ffmpeg_command.run(capture_stdout=True, capture_stderr=True)

        # *** Debugging Message *** #
        print(f"Cropped video saved to: {output_file}")

    except ffmpeg.Error as e:
        # *** Debugging Message *** #
        print(f"FFMPEG error: {e.stderr.decode('utf-8')}")

    except Exception as e:
        # *** Debugging Message *** #
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
    """Smoother interpolation using cubic easing."""
    # Improved cubic bezier curve
    t2 = t * t
    t3 = t2 * t
    # Use a more gradual curve for smoother motion
    return pos1 * (1 - t3) + pos2 * t3

# ----------------------------------------------------------------------------
# Detect a face in a video and crop the video around the face
# ----------------------------------------------------------------------------


def detect_face_and_crop(video_path, output_path, start_time, end_time):
    try:
        # Validate duration and adjust times
        probe = ffmpeg.probe(video_path) 
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        total_duration = float(probe['format']['duration'])
        start_time = max(0, min(float(start_time), total_duration))
        end_time = max(start_time, min(float(end_time), total_duration))
        duration = end_time - start_time
        if duration <= 0:
            raise ValueError("Invalid duration: end_time must be greater than start_time")
        if duration > 59.5:
            raise ValueError("Video duration cannot exceed 59.5 seconds")
        
        # Get video dimensions and frame rate
        in_width = int(video_info['width'])
        in_height = int(video_info['height'])
        fps = eval(video_info['r_frame_rate'])
    
    except ffmpeg.Error as e:
        print(f"Error probing video file: {str(e)}")
        raise
    except Exception as e:
        print(f"Error validating video parameters: {str(e)}")
        raise

    # Define buffer time and calculate buffered clip times
    buffer_time = 2  # seconds
    buffered_start = max(0, start_time - buffer_time)
    buffered_end = min(total_duration, end_time + buffer_time)
    buffered_duration = buffered_end - buffered_start

    # Process using a temporary buffered video clip
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract temporary buffered clip
        temp_clip = os.path.join(temp_dir, 'temp_clip.mp4')
        try:
            (
                ffmpeg
                .input(video_path, ss=buffered_start, t=buffered_duration)
                .output(temp_clip, codec='copy')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
        except ffmpeg.Error as e:
            print(f"Error extracting temporary clip: {e.stderr.decode('utf-8')}")
            raise

        # Adjust relative start time for the buffered clip
        rel_start_time = start_time - buffered_start

        # Initialize video capture on the temporary clip
        cap = cv2.VideoCapture(temp_clip)
        positions = []
        last_valid_crop = None
        crop_width = int(in_height * (9/16))
        crop_height = in_height
        batch_size = 4 if use_cuda else 1  # Process multiple frames at once if using GPU

        frame_count = 0
        frames_batch = []
        try:
            while cap.isOpened():
                frames_batch.clear()
                # Read batch_size frames
                for _ in range(batch_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames_batch.append(prepare_frame(frame))
                
                if not frames_batch:
                    break

                frame_count += len(frames_batch)
                if frame_count % 100 == 0:
                    print(f"Processing frame {frame_count}")
                    # Force garbage collection periodically
                    gc.collect()
                    if use_cuda:
                        torch.cuda.empty_cache()

                # Process batch of frames
                with torch.no_grad():
                    if len(frames_batch) == batch_size:
                        batch_boxes, _ = mtcnn.detect(frames_batch)
                    else:
                        batch_boxes = []
                        for frame in frames_batch:
                            boxes, _ = mtcnn.detect(frame)
                            batch_boxes.append(boxes)

                # Process each frame's detection results
                for boxes in (batch_boxes if isinstance(batch_boxes, list) else [boxes for boxes in batch_boxes]):
                    if boxes is not None and len(boxes) > 0:
                        if isinstance(boxes, torch.Tensor):
                            boxes = boxes.detach().cpu().numpy()
                        
                        valid_boxes = []
                        for box in boxes:
                            try:
                                box_arr = np.array(box, dtype=np.float32)
                            except Exception as e:
                                continue
                            if not (np.isnan(box_arr).any() or np.isinf(box_arr).any()):
                                valid_boxes.append(box)
                        
                        if valid_boxes:
                            valid_boxes.sort(key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
                            best_box = valid_boxes[0]
                            x1, y1, x2, y2 = best_box
                            w = x2 - x1
                            h = y2 - y1
                            face_center_x = x1 + w/2

                            best_x = max(0, min(face_center_x - crop_width/2, in_width - crop_width))
                            if last_valid_crop:
                                max_movement = crop_width * 0.015
                                delta_x = best_x - last_valid_crop[0]
                                if abs(delta_x) > max_movement:
                                    best_x = last_valid_crop[0] + (max_movement if delta_x > 0 else -max_movement)
                            last_valid_crop = (best_x, 0)
                            positions.append((best_x, 0, crop_width, crop_height))
                        else:
                            crop_x = last_valid_crop[0] if last_valid_crop else (in_width - crop_width) // 2
                            positions.append((crop_x, 0, crop_width, crop_height))
                    else:
                        crop_x = last_valid_crop[0] if last_valid_crop else (in_width - crop_width) // 2
                        positions.append((crop_x, 0, crop_width, crop_height))
        finally:
            cap.release()
            if use_cuda:
                torch.cuda.empty_cache()
            gc.collect()

        print(f"Total frames processed: {frame_count}")

        # Smooth transitions between positions
        smoothed_positions = []
        for i in range(len(positions) - 1):
            start_pos = positions[i]
            end_pos = positions[i + 1]
            smoothed_positions.extend(generate_tween_frames(start_pos, end_pos, int(fps)))
        smoothed_positions.append(positions[-1])

        # Process buffered clip with detected positions
        try:
            # Extract audio from the temporary clip relative to the desired start time
            audio_file = os.path.join(temp_dir, 'audio.aac')
            (
                ffmpeg
                .input(temp_clip, ss=rel_start_time, t=duration)
                .output(audio_file, acodec='aac', audio_bitrate='192k')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )

            segments = []
            segment_duration = duration / len(smoothed_positions)

            for i, (crop_x, crop_y, crop_width, crop_height) in enumerate(smoothed_positions):
                # Adjust segment start relative to the buffered clip
                segment_start = rel_start_time + (i * segment_duration)
                segment_path = os.path.join(temp_dir, f'segment_{i:03d}.mp4')
                segments.append(segment_path)

                try:
                    (
                        ffmpeg
                        .input(temp_clip, ss=segment_start, t=segment_duration)
                        .filter('crop', crop_width, crop_height, crop_x, crop_y)
                        .filter('setpts', 'PTS-STARTPTS')
                        .filter('fps', fps=fps, round='up')
                        .filter('scale', 1080, 1920)
                        .output(segment_path, vcodec='libx264', crf=23, preset='fast', r=fps)
                        .overwrite_output()
                        .run(capture_stdout=True, capture_stderr=True, quiet=True)
                    )
                except ffmpeg.Error as e:
                    print(f"Error processing segment {i}: {e.stderr.decode('utf-8')}")
                    raise

            list_file = os.path.join(temp_dir, 'segments.txt')
            with open(list_file, 'w') as f:
                for segment in segments:
                    f.write(f"file '{os.path.abspath(segment)}'\n")

            video_file = os.path.join(temp_dir, 'video.mp4')
            (
                ffmpeg
                .input(list_file, format='concat', safe=0)
                .output(video_file, c='copy')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )

            video_in = ffmpeg.input(video_file)
            audio_in = ffmpeg.input(audio_file)
            (
                ffmpeg
                .output(video_in, audio_in, output_path, vcodec='copy', acodec='copy', movflags='+faststart', r=fps)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )

        except ffmpeg.Error as e:
            print(f"Error processing video: {e.stderr.decode('utf-8')}")
            raise
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            raise

        print(f"Successfully created video: {output_path}")
