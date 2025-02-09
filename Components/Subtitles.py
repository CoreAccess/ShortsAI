# Import the VideoFileClip class from the moviepy library
from moviepy.video.io.VideoFileClip import VideoFileClip
# Import the CompositeVideoClip class for combining video clips
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
# Import the ImageClip class for creating video clips from images
from moviepy.video.VideoClip import ImageClip
# Import the Image, ImageDraw, and ImageFont classes from the PIL library for image manipulation
from PIL import Image, ImageDraw, ImageFont
import textwrap  # Import the textwrap module for wrapping text
import numpy as np  # Import the numpy library for numerical operations
import ffmpeg  # Import the ffmpeg library for video processing
from Components.Helpers import format_timestamp  # NEW IMPORT


def add_subtitles(video_path, subtitles, output_path):
    # Define a function called add_subtitles that takes the video path, subtitles, and output path as input
    """
    Overlay subtitles on the video.
    For each subtitle, the text is wrapped and drawn over a semi-transparent background.
    """
    video = VideoFileClip(video_path)
    # Create a VideoFileClip object from the given video path
    subtitle_clips = []
    # Initialize an empty list to store the subtitle clips
    font_path = "arialbd.ttf"
    # Set the default font path to arialbd.ttf
    font_size = int(48 * 2.5)
    # Set the font size
    outline_color = "black"
    # Set the outline color for the text
    text_color = "white"
    # Set the text color

    if not __import__("os").path.exists(font_path):
        # Check if the specified font path does not exist
        alternate_font = "C:/Windows/Fonts/arialbd.ttf"
        # Define an alternate font path
        if __import__("os").path.exists(alternate_font):
            # Check if the alternate font path exists
            font_path = alternate_font
            # If it exists, use the alternate font path
        else:
            # If neither font path exists
            from streamlit import warning
            # Import the warning function from streamlit
            warning("Preferred font not found; falling back to default font.")
            # Display a warning message in the streamlit app
            font_path = None
            # Set the font path to None, which will use the default font

    for subtitle in subtitles:
        # Iterate over each subtitle in the subtitles list
        orig_start, orig_end, text = subtitle
        # Unpack the subtitle tuple into start time, end time, and text
        wrapped = textwrap.wrap(text, width=40)
        # Wrap the text to a maximum width of 40 characters
        if len(wrapped) > 3:
            # If the number of wrapped lines is greater than 3
            wrapped = wrapped[:3]
            # Truncate the wrapped lines to the first 3 lines
        text = "\n".join(wrapped)
        # Join the wrapped lines with newline characters
        duration = orig_end - orig_start
        # Calculate the duration of the subtitle
        if duration <= 0:
            # If the duration is less than or equal to 0
            duration = 1.0
            # Set the duration to 1.0
        img_height = 125
        # Set the height of the image
        img = Image.new('RGBA', (video.w, img_height), (0, 0, 0, 128))
        # Create a new RGBA image with a semi-transparent background
        d = ImageDraw.Draw(img)
        # Create a drawing object for the image
        try:
            # Try to load the specified font
            font = ImageFont.truetype(
                font_path, font_size) if font_path else ImageFont.load_default()
            # Load the font from the specified font path, or use the default font if the font path is None
        except OSError:
            # If an OSError occurs while loading the font
            from streamlit import warning
            # Import the warning function from streamlit
            warning("Error loading chosen font; using default.")
            # Display a warning message in the streamlit app
            font = ImageFont.load_default()
            # Load the default font

        text_bbox = d.multiline_textbbox((0, 0), text, font=font, spacing=4)
        # Get the bounding box of the text
        text_width = text_bbox[2] - text_bbox[0]
        # Calculate the width of the text
        text_height = text_bbox[3] - text_bbox[1]
        # Calculate the height of the text
        x = (video.w - text_width) // 2
        # Calculate the x position of the text
        y = (img_height - text_height) // 2
        # Calculate the y position of the text

        for dx in range(-2, 3):
            # Iterate over the x offsets
            for dy in range(-2, 3):
                # Iterate over the y offsets
                if dx == 0 and dy == 0:
                    # If both offsets are 0, skip this iteration
                    continue
                d.multiline_text((x+dx, y+dy), text, font=font,
                                 fill=outline_color, spacing=4)
                # Draw the text with the outline color
        d.multiline_text((x, y), text, font=font, fill=text_color, spacing=4)
        # Draw the text with the text color
        y_position = int(video.h * 0.75 - img_height / 2)
        # Calculate the y position of the subtitle
        txt_clip = ImageClip(np.array(img)).set_duration(duration).set_start(
            orig_start).set_position(('center', y_position))
        # Create an ImageClip object from the image, set the duration, start time, and position
        subtitle_clips.append(txt_clip)
        # Add the subtitle clip to the list of subtitle clips
    video = CompositeVideoClip([video] + subtitle_clips)
    # Create a CompositeVideoClip object from the video and the subtitle clips
    video.write_videofile(output_path, codec='libx264',
                          bitrate="5000k", fps=video.fps)
    # Write the video to the specified output path with the specified codec, bitrate, and fps
    video.close()
    # Close the video file


def write_srt(subtitles, srt_file_path):
    """
    Create an SRT subtitle file.
    Each subtitle entry's text is wrapped to a maximum of 40 characters per line over up to 3 lines.
    """
    max_chars = 40
    max_lines = 3
    index = 1
    with open(srt_file_path, 'w', encoding='utf-8') as f:
        for start, end, text in subtitles:
            if end - start < 1.0:
                end = start + 1.0
            lines = textwrap.wrap(text, width=max_chars)[:max_lines]
            wrapped_text = "\n".join(lines)
            f.write(f"{index}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{wrapped_text}\n\n")
            index += 1


def burn_subtitles(video_path, srt_path, output_path):
    """
    Burn subtitles directly onto the video using ffmpeg.
    The styling here is minimal – plain Impact font in white.
    """
    srt_path_fixed = srt_path.replace("\\", "/")
    try:
        ffmpeg.input(video_path).output(
            output_path,
            vf=f"subtitles='{srt_path_fixed}':force_style='FontName=Impact'",
            **{'c:a': 'copy', 'c:v': 'libx264'}
        ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        error_message = e.stderr.decode(
            'utf-8') if e.stderr else "No stderr output."
        print("FFmpeg error details:", error_message)
        raise
