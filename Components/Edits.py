from moviepy.video.io.VideoFileClip import VideoFileClip
import streamlit as st
import cv2
import numpy as np


# ----------------------------------------------------------------------------
# Extract audio from a video file
# ----------------------------------------------------------------------------
def extractAudio(video_path, audio_path):
    try:  # Try to execute the following code block

        # Create a VideoFileClip object from the given video path
        video_clip = VideoFileClip(video_path)

        # Extract the audio from the video clip and save it to the specified audio path
        video_clip.audio.write_audiofile(audio_path)

        # Close the video clip to release resources
        video_clip.close()

        # Display a message indicating that the audio extraction is complete
        st.write(f"Extracted Audio To: {audio_path}")

        # Return the path to the extracted audio file
        return audio_path

    # If an exception occurs during the try block, execute this code block
    except Exception as e:
        # Write an error message
        st.write(f"An error occurred while extracting audio: {e}")

        # Check if the video_clip variable exists in the local scope
        if 'video_clip' in locals():
            # If it exists, close the video clip to release resources
            video_clip.close()

        # Return None to indicate that the audio extraction failed
        return None


# ----------------------------------------------------------------------------
# Crop a video file
# ----------------------------------------------------------------------------
def crop_video(input_file, output_file, start_time, end_time):
    # Open the input video file as a VideoFileClip object using a context manager
    with VideoFileClip(input_file) as video:
        # Create a subclip of the video from the start time to the end time
        cropped_video = video.subclip(start_time, end_time)

        # Write the cropped video to the specified output file using the libx264 codec
        cropped_video.write_videofile(output_file, codec='libx264')


# ----------------------------------------------------------------------------
# Detect a face in a video and crop the video around the face
# ----------------------------------------------------------------------------
def detect_face_and_crop(video_path, output_path, start_time, end_time):
    try:
        with VideoFileClip(video_path) as video:
            subclip = video.subclip(start_time, min(end_time, start_time + 59))
            width, height = subclip.size
            target_aspect_ratio = 9 / 16
            target_width = height * target_aspect_ratio

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            face_positions = []
            num_frames_to_analyze = min(
                int(subclip.fps * 2), int(subclip.duration * subclip.fps))
            for t in np.linspace(0, min(2, subclip.duration), num=num_frames_to_analyze):
                frame = subclip.get_frame(t)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(faces) > 0:
                    x, y, w, h_face = faces[0]
                    center_x = x + w // 2
                    face_positions.append(center_x)

            avg_center_x = int(np.mean(face_positions)
                               ) if face_positions else (width // 2)
            left = max(0, avg_center_x - target_width // 2)
            right = min(width, avg_center_x + target_width // 2)

            # Set fixed output dimensions (portrait: 1080x1920)
            output_width, output_height = 1080, 1920

            def crop_frame(frame):
                cropped = frame[:, int(left):int(right)]
                return cv2.resize(cropped, (output_width, output_height))

            cropped_subclip = subclip.fl_image(crop_frame)
            cropped_subclip.write_videofile(
                output_path, codec='libx264', bitrate="5000k", fps=subclip.fps)
            cropped_subclip.close()
            st.write(f"Cropped video saved to: {output_path}")
    except Exception as e:
        st.error(f"Face detection/cropping error: {e}")
        raise
