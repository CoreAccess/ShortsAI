from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.VideoClip import ImageClip
from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np
import ffmpeg
from Components.Helpers import format_timestamp
from streamlit import warning

# ----------------------------------------------------------------------------
# Overlay subtitles on the video.
# ----------------------------------------------------------------------------


def add_subtitles(video_path, subtitles, output_path):
    # Create a VideoFileClip object from the given video path
    video = VideoFileClip(video_path)

    # Initialize an empty list to store the subtitle clips
    subtitle_clips = []

    # Set the default font path to arialbd.ttf
    font_path = "arialbd.ttf"

    # Set the font size
    font_size = int(48 * 2.5)

    # Set the outline color for the text
    outline_color = "black"

    # Set the text color
    text_color = "white"

    # Check if the specified font path does not exist
    if not __import__("os").path.exists(font_path):
        # Define an alternate font path
        alternate_font = "C:/Windows/Fonts/arialbd.ttf"

        # Check if the alternate font path exists
        if __import__("os").path.exists(alternate_font):
            # If it exists, use the alternate font path
            font_path = alternate_font

        else:  # If neither font path exists

            # Display a warning message
            warning("Preferred font not found; falling back to default font.")

            # Set the font path to None, which will use the default font
            font_path = None

    # Iterate over each subtitle in the subtitles list
    for subtitle in subtitles:
        # Unpack the subtitle tuple into start time, end time, and text
        orig_start, orig_end, text = subtitle

        # Wrap the text to a maximum width of 40 characters
        wrapped = textwrap.wrap(text, width=40)

        # If the number of wrapped lines is greater than 3
        if len(wrapped) > 3:
            # Truncate the wrapped lines to the first 3 lines
            wrapped = wrapped[:3]

        # Join the wrapped lines with newline characters
        text = "\n".join(wrapped)

        # Calculate the duration of the subtitle
        duration = orig_end - orig_start

        # If the duration is less than or equal to 0
        if duration <= 0:
            # Set the duration to 1.0
            duration = 1.0

        # Set the height of the image
        img_height = 125

        # Create a new RGBA image with a semi-transparent background
        img = Image.new('RGBA', (video.w, img_height), (0, 0, 0, 128))

        # Create a drawing object for the image
        d = ImageDraw.Draw(img)

        # Try to load the specified font
        try:
            # Load the font from the specified font path
            font = ImageFont.truetype(
                font_path, font_size) if font_path else ImageFont.load_default()

        except OSError:  # If an OSError occurs while loading the font

            # Display a warning message
            warning("Error loading chosen font; using default.")

            # Load the default font
            font = ImageFont.load_default()

        # Get the bounding box of the text
        text_bbox = d.multiline_textbbox((0, 0), text, font=font, spacing=4)

        # Calculate the width of the text
        text_width = text_bbox[2] - text_bbox[0]

        # Calculate the height of the text
        text_height = text_bbox[3] - text_bbox[1]

        # Calculate the x position of the text
        x = (video.w - text_width) // 2

        # Calculate the y position of the text
        y = (img_height - text_height) // 2

        # Iterate over the x offsets
        for dx in range(-2, 3):
            # Iterate over the y offsets
            for dy in range(-2, 3):
                # If both offsets are 0, skip this iteration
                if dx == 0 and dy == 0:
                    continue

                # Draw the text with the outline color
                d.multiline_text((x+dx, y+dy), text, font=font,
                                 fill=outline_color, spacing=4)

        # Draw the text with the text color
        d.multiline_text((x, y), text, font=font, fill=text_color, spacing=4)

        # Calculate the y position of the subtitle
        y_position = int(video.h * 0.75 - img_height / 2)

        # Create an ImageClip object from the image
        txt_clip = ImageClip(np.array(img)).set_duration(duration).set_start(
            orig_start).set_position(('center', y_position))

        # Add the subtitle clip to the list of subtitle clips
        subtitle_clips.append(txt_clip)

    # Create a CompositeVideoClip object from the video and the subtitle clips
    video = CompositeVideoClip([video] + subtitle_clips)

    # Write the video to the specified output path
    video.write_videofile(output_path, codec='libx264',
                          bitrate="5000k", fps=video.fps)

    # Close the video file
    video.close()

# ----------------------------------------------------------------------------
# Create an SRT subtitle file.
# ----------------------------------------------------------------------------


def write_srt(subtitles, srt_file_path):
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

# ----------------------------------------------------------------------------
# Burn subtitles directly onto the video using ffmpeg.
# ----------------------------------------------------------------------------


def burn_subtitles(video_path, srt_path, output_path):
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
