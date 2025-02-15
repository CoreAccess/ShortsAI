import os
import ffmpeg
import pysubs2
from components.helpers import normalize_path, get_video_duration

# Dynamically determine the path to the subscribe image
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
subscribe_img_path = os.path.join(PROJECT_DIR, "static", "assets", "youtube-subscribe.png")

# ----------------------------------------------------------------------------
# Create an ASS subtitle file.
# ----------------------------------------------------------------------------

def write_ass(subtitles, ass_file_path, video_path):
    ass_file_path = normalize_path(ass_file_path)
    
    subs = pysubs2.SSAFile()
    
    # Set video resolution explicitly for portrait mode
    width, height = 1080, 1920

    # Set up required ASS script info
    subs.info = {
        'Title': 'Styled Subtitles',
        'ScriptType': 'v4.00+',
        'WrapStyle': '0',
        'ScaledBorderAndShadow': 'yes',
        'PlayResX': str(width),
        'PlayResY': str(height),
        'Collisions': 'Normal'
    }

    # Define the default style
    subs.styles['Default'] = pysubs2.SSAStyle(
        fontname="impact",
        fontsize=80,
        primarycolor=pysubs2.Color(255, 255, 255),
        outlinecolor=pysubs2.Color(0, 0, 0),
        backcolor=pysubs2.Color(0, 0, 0, 100),
        bold=-1,
        italic=0,
        underline=0,
        strikeout=0,
        scalex=100, # Adjust these to change font size
        scaley=100, # Adjust these to change font size
        spacing=0.8,
        angle=0,
        borderstyle=1,
        outline=2.5,
        shadow=0.5,
        alignment=2,  # 2 = bottom-center alignment
        marginl=10,
        marginr=10,
        marginv=500,  # Increase margin from bottom
        encoding=1
    )

    print(f"Creating subtitles with {len(subtitles)} words")
    
    subtitles = sorted(subtitles, key=lambda x: x[0])
    
    for i, (word_start, word_end, current_word, sent_num) in enumerate(subtitles):
        # Display 4 words at a time
        window = 4
        if len(subtitles) <= window:
            start_idx = 0
            end_idx = len(subtitles)
        else:
            offset = (window - 1) // 2  # For window=4, offset will be 1
            start_idx = max(0, i - offset)
            if start_idx + window > len(subtitles):
                start_idx = len(subtitles) - window
            end_idx = start_idx + window

        # Define the color cycle: white, green, yellow.
        # Red = &HFF0000&, Green = &H00FF00&, Yellow = &H00FFFF&
        color_cycle = ["&HFF0000&", "&H00FF00&", "&H00FFFF&"]

        highlighted_parts = []

        for j in range(start_idx, end_idx):
            word = subtitles[j][2].upper()
            # Only highlight the current word.
            if j == i:
                color = color_cycle[j % len(color_cycle)]
                highlighted_parts.append(f"{{\\1c{color}\\bord2.5\\shad0.5}}{word}\r")
            else:
                highlighted_parts.append(word)

        highlighted_text = " ".join(highlighted_parts)

        if i == 0:
            print(f"First subtitle: {highlighted_text}")
            print(f"Time: {word_start:.2f} - {word_end:.2f}")

        # Extend event display by 1 extra second (1000 ms) but ensure we don't overlap with next event
        current_start_ms = word_start * 1000
        current_end_ms = word_end * 1000
        extended_end_ms = current_end_ms + 1000  # add 1 second
        if i < len(subtitles) - 1:
            next_start_ms = subtitles[i + 1][0] * 1000
            extended_end_ms = min(extended_end_ms, next_start_ms)

        event = pysubs2.SSAEvent(
            start=pysubs2.make_time(ms=current_start_ms),
            end=pysubs2.make_time(ms=extended_end_ms),
            text=f"{highlighted_text} ({current_word})",
            style='Default'
        )
        subs.events.append(event)

    #print(f"Generated {len(subs.events)} subtitle events")
    print(f"Saving subtitles to: {ass_file_path}")
    
    # Save with explicit encoding
    subs.save(ass_file_path, encoding='utf-8-sig')
    
    print("Subtitles saved successfully")

# ----------------------------------------------------------------------------
# Burn subtitles directly onto the video using ffmpeg.
# ----------------------------------------------------------------------------

def burn_subtitles(video_path, ass_path, output_path):
    print(f"Starting subtitle burn process...")
    print(f"Input video: {video_path}")
    print(f"Subtitle file: {ass_path}")
    print(f"Output path: {output_path}")
    
    # Normalize all paths to use forward slashes
    video_path = normalize_path(video_path)
    ass_path = normalize_path(ass_path)
    output_path = normalize_path(output_path)
    subscribe_img_path_fixed = normalize_path(subscribe_img_path)
    
    try:
        # Get video duration for overlay timing
        video_duration = get_video_duration(video_path)
        overlay_start = video_duration - 15.0
        print(f"Video duration: {video_duration}, overlay will start at: {overlay_start}")

        # Create input streams
        video = ffmpeg.input(video_path)
        overlay = ffmpeg.input(subscribe_img_path_fixed)

        # Apply subtitle filter and overlay in one step
        ass_video = video.filter('ass', filename=ass_path)
        final_video = ffmpeg.overlay(
            ass_video,
            overlay,
            x='main_w-overlay_w*1.4-10',
            y=f"if(gte(t,{overlay_start}),if(lt(t,{overlay_start}+1),10-70*sin(PI*(t-{overlay_start})),10),10)",
            enable=f"gte(t,{overlay_start}-1)"
        )

        print("Starting FFmpeg process...")
        
        # Check if the input video has an audio stream
        probe = ffmpeg.probe(video_path)
        has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])

        # Create the output with both video and audio if audio stream exists
        if has_audio:
            stream = ffmpeg.output(final_video, video.audio, output_path, acodec='copy', vcodec='libx264', preset='fast', crf=23, movflags='+faststart', threads=4)
        else:
            stream = ffmpeg.output(final_video, output_path, vcodec='libx264', preset='fast', crf=23, movflags='+faststart', threads=4)

        # Run the ffmpeg command
        #print(f"FFmpeg command: {stream.compile()}")  # Debug: print the FFmpeg command
        stream.overwrite_output().run(capture_stdout=True, capture_stderr=True)
        print(f"FFmpeg process completed.")  # Debug: print when FFmpeg process is completed
        
        print(f"Subtitles burned successfully, output saved to: {output_path}")
 
    except ffmpeg.Error as e:
        err = e.stderr.decode('utf-8') if e.stderr else "No stderr output."
        print(f"FFmpeg error during subtitle burn: {err}")
        raise

    # Re-mux the final video to fix any metadata issues if needed
    try:
        remuxed_path = output_path.replace(".mp4", "_fixed.mp4")
        (
            ffmpeg
            .input(output_path)
            .output(remuxed_path, codec="copy", **{'map_metadata': '-1', 'movflags': '+faststart'})
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        os.replace(remuxed_path, output_path)
        print("Video metadata fixed by re-muxing.")
    except ffmpeg.Error as e:
        err = e.stderr.decode('utf-8') if e.stderr else "No stderr output."
        print(f"FFmpeg re-mux error: {err}")
        # Continue even if re-muxing fails
