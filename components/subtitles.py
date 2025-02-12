import os
import textwrap
import ffmpeg
from components.helpers import format_timestamp, get_video_duration

# Dynamically determine the path to the subscribe image
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
subscribe_img_path = os.path.join(PROJECT_DIR, "static", "assets", "youtube-subscribe.png")

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
            # Prepend the override tag for center alignment
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

    # ------------------------------------------------------------------------
    # Step 1: Get video duration.
    # ------------------------------------------------------------------------
    video_duration = get_video_duration(video_path)
    overlay_start = video_duration - 15.0  # Start overlay 15 seconds before end
    print(
        f"[DEBUG] Video duration: {video_duration}, overlay will start at: {overlay_start}")

    # ------------------------------------------------------------------------
    # Step 2: Define animated overlay parameters.
    # ------------------------------------------------------------------------
    # Increase the size of the overlay by 40%
    scale = 1.4
    scaled_width = f"iw*{scale}"
    scaled_height = f"ih*{scale}"

    # Align to the top right corner with a 10-pixel margin
    x = f"main_w-overlay_w*{scale}-10"
    y_static = 10

    # Reduce the bounce by 30%
    bounce_reduction = 0.7
    bounce_amplitude = 100 * bounce_reduction

    # Define the bounce animation
    y = f"if(gte(t,{overlay_start}),if(lt(t,{overlay_start}+1),{y_static}-{bounce_amplitude}*sin(PI*(t-{overlay_start})),{y_static}),-overlay_h)"
    print(f"[DEBUG] Using subscribe image file: {subscribe_img_path}")

    # ------------------------------------------------------------------------
    # Step 3: Combine subtitle burning and overlay in a single FFmpeg command.
    # ------------------------------------------------------------------------
    try:
        # Define the complex filtergraph
        input_video = ffmpeg.input(video_path)
        input_overlay = ffmpeg.input(subscribe_img_path)

        # Subtitle filter
        subtitles = input_video.filter(
            'subtitles', srt_path_fixed, force_style='FontName=Impact,Alignment=2,MarginV=65'
        )

        # Scale the overlay
        scaled_overlay = input_overlay.filter(
            'scale', w=scaled_width, h=scaled_height)

        # Overlay filter
        overlay = subtitles.overlay(
            scaled_overlay,
            x=x,
            y=y,
            enable=f'gte(t,{overlay_start}-1)'
        )

        # Output
        (
            overlay
            .output(output_path, vcodec='libx264', acodec='copy', map='0:a')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(
            f"[DEBUG] Subtitles and overlay applied; output saved to: {output_path}")

    except ffmpeg.Error as e:
        err = e.stderr.decode('utf-8') if e.stderr else "No stderr output."
        print("[DEBUG] FFmpeg error during processing:", err)
        raise

    # ------------------------------------------------------------------------
    # Step 4: (Optional) Re-mux the final video to fix metadata issues.
    # ------------------------------------------------------------------------
    # For testing purposes, comment out the re-mux step until you verify the overlay.
    # Sometimes remuxing with 'codec=copy' can discard changes applied during encoding.
    remuxed_path = output_path.replace(".mp4", "_fixed.mp4")
    try:
        (
            ffmpeg
            .input(output_path)
            .output(
                remuxed_path,
                codec="copy",
                **{'map_metadata': '-1', 'movflags': '+faststart', 'reset_timestamps': '1'}
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        import os
        os.replace(remuxed_path, output_path)
        print("[DEBUG] Subtitled video metadata fixed by re-muxing.")
    except ffmpeg.Error as e:
        # This error will show if re-muxing fails.
        err = e.stderr.decode('utf-8') if e.stderr else "No stderr output."
        print(f"[DEBUG] FFmpeg re-mux error (burn_subtitles): {err}")
        # If re-muxing isn't required, you can comment this block out.
