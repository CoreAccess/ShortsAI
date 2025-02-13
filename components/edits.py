import cv2
import numpy as np
import ffmpeg
import os

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
# Helper function to detect faces in a frame
# ----------------------------------------------------------------------------


def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

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
# Detect a face in a video and crop the video around the face
# ----------------------------------------------------------------------------


def detect_face_and_crop(video_path, output_path, start_time, end_time):
    # Calculate overall video dimensions.
    in_width, in_height = get_video_dimensions(video_path)
    if not in_width or not in_height:
        raise ValueError("Could not determine video dimensions.")

    duration = end_time - start_time
    target_ratio = 9 / 16

    # Initialize face cascade.
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Helper: detect the primary face center at a given time.
    def get_primary_face_center(time_sec):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
        ret, frame = cap.read()
        cap.release()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) == 0:
                return None
            # Focus on the largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            avg_face_x = largest_face[0] + largest_face[2] // 2
            avg_face_y = largest_face[1] + largest_face[3] // 2
            return avg_face_x, avg_face_y
        return None

    # Sample face centers at every 20% interval
    import numpy as np
    sample_percentages = [0.2, 0.4, 0.6, 0.8]
    sample_times = [start_time + duration * p for p in sample_percentages]
    samples = [get_primary_face_center(t) for t in sample_times]

    # Filter out None values
    samples = [s for s in samples if s is not None]

    if not samples:
        raise ValueError("No faces detected in the video segment.")

    # Use the first detected face as the primary face
    primary_face_x, primary_face_y = samples[0]

    # Helper lambda for clamping values.
    def clamp(x, lo, hi): return max(lo, min(x, hi))

    # Determine crop rectangle based on primary face
    if (in_width / in_height) >= target_ratio:
        # Horizontal branch: crop width based on in_height.
        crop_width = int(in_height * target_ratio)
        crop_y = 0
        crop_x = clamp(primary_face_x - crop_width // 2, 0, in_width - crop_width)
    else:
        # Vertical branch: crop height computed from in_width.
        crop_height = int(in_width * (16/9))
        crop_x = 0
        crop_y = clamp(primary_face_y - crop_height // 2, 0, in_height - crop_height)

    try:
        (
            ffmpeg
            .input(video_path, ss=start_time, to=end_time)
            .filter('crop',
                    w=crop_width if (
                        in_width/in_height) >= target_ratio else in_width,
                    h=crop_height if (
                        in_width/in_height) < target_ratio else in_height,
                    x=crop_x,
                    y=crop_y)
            .filter('scale', 1080, 1920)
            .output(output_path,
                    vcodec='libx264',
                    acodec='aac',
                    audio_bitrate='192k',
                    preset='fast', crf=23,
                    **{'map': '0:a?'}
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"Cropped video saved to: {output_path}")
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode('utf-8')}")
        raise

    # Re-mux the file to fix metadata using additional flags (reset timestamps)
    remuxed_path = output_path.replace(".mp4", "_fixed.mp4")
    try:
        (
            ffmpeg
            .input(output_path)
            .output(remuxed_path,
                    codec="copy",
                    **{'map_metadata': '-1', 'movflags': '+faststart', 'reset_timestamps': '1'})
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        os.replace(remuxed_path, output_path)
        print("Video metadata fixed by resetting timestamps via re-mux.")
    except ffmpeg.Error as e:
        print(
            f"FFmpeg re-mux error (reset timestamps): {e.stderr.decode('utf-8')}")
