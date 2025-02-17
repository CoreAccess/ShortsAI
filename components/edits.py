import cv2
import numpy as np
import torch
import librosa
import os
from moviepy.editor import VideoFileClip, AudioFileClip, VideoClip, clips_array, vfx
import subprocess
from typing import Tuple, List, Dict, Optional
import mediapipe as mp
from collections import deque
from contextlib import contextmanager
import tempfile

class VideoProcessor:
    """Main class for processing videos into YouTube Shorts format"""
    
    def __init__(self):
        # Initialize MediaPipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.85
        )
        
        # Initialize device for processing
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Speaker detection parameters
        self.speaking_window = 15  # frames
        self.speaking_history = deque(maxlen=self.speaking_window)
         
        # Face tracking parameters
        self.tracking_window = 30  # frames
        self.position_history = deque(maxlen=self.tracking_window)
        
        # Set up paths
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.temp_dir = os.path.join(self.project_dir, "temp_files")
        self.finished_dir = os.path.join(self.project_dir, "finished_videos")
        
        # Ensure directories exist
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.finished_dir, exist_ok=True)
        
    def detect_face_and_crop(self, video_path: str, start: float, end: float) -> str:
        """
        Main entry point for processing a video segment into YouTube Shorts format.
        
        Args:
            video_path: Path to the source video
            start: Start timestamp in seconds
            end: End timestamp in seconds
            
        Returns:
            Path to the processed video file in finished_videos directory
        """
        try:
            print("\n=== Starting Video Processing Pipeline ===")
            duration = end - start
            file_hash = os.path.splitext(os.path.basename(video_path))[0]
            
            # Define output paths
            temp_clip_path = os.path.join(self.temp_dir, f"{file_hash}_temp_clip_{start:.2f}_{end:.2f}.mp4")
            debug_output_path = os.path.join(self.finished_dir, f"{file_hash}_debug_{start:.2f}_{end:.2f}.mp4")
            final_output_path = os.path.join(self.finished_dir, f"{file_hash}_short_{start:.2f}_{end:.2f}.mp4")
            
            print("Step 1: Extracting working clip...")
            # Step 1: Extract working clip
            temp_clip_path = self._extract_temp_clip(
                video_path, start, duration, temp_clip_path
            )
            
            print("Step 2: Pre-analyzing video...")
            # Step 2: Pre-analyze the video for speech and face positions
            analyzer = SpeechAndFaceAnalyzer()
            cap = cv2.VideoCapture(temp_clip_path)
            fps = int(cap.get(5))  # cv2.CAP_PROP_FPS is 5
            cap.release()
            
            analysis = analyzer.analyze_video(temp_clip_path, fps)
            
            print("Step 3: Processing frames with analysis...")
            # Step 3: Process the video frames using the analysis data
            processed_frames = self._process_video_with_analysis(temp_clip_path, analysis)
            
            if not processed_frames:
                raise ValueError("No frames were processed")
            
            print("Step 4: Generating debug visualization...")
            # Step 4: Generate debug visualization
            debug_path = self._create_debug_video(
                temp_clip_path,
                processed_frames,
                debug_output_path
            )
            
            print("Step 5: Creating final cropped video...")
            # Step 5: Create final cropped video
            final_path = self._create_final_video(
                temp_clip_path,
                processed_frames,
                final_output_path
            )
            
            # Clean up temporary files
            if os.path.exists(temp_clip_path):
                os.remove(temp_clip_path)
                print("Cleaned up temporary files")
            
            print("Video processing complete!")
            return final_output_path
                
        except Exception as e:
            print(f"Error in detect_face_and_crop: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def _extract_temp_clip(self, video_path: str, start: float, duration: float, output_path: str) -> str:
        """Extract working segment from the main video"""
        try:
            # Use MoviePy to extract the segment
            with VideoFileClip(video_path) as video:
                clip = video.subclip(start, start + duration)
                clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
                
            return output_path
            
        except Exception as e:
            print(f"Error extracting temp clip: {str(e)}")
            raise
            
    def _process_video(self, video_path: str) -> List[Dict]:
        """Process video frames for face detection and speaker detection"""
        processed_frames = []
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(5))  # cv2.CAP_PROP_FPS is 5
        frame_width = int(cap.get(3))  # cv2.CAP_PROP_WIDTH is 3
        frame_height = int(cap.get(4))  # cv2.CAP_PROP_HEIGHT is 4
        
        # Calculate target width to ensure it's divisible by 2
        raw_target_width = int(frame_height * 9 / 16)
        target_width = raw_target_width if raw_target_width % 2 == 0 else raw_target_width + 1
        
        # Initialize speaker detector
        speaker_detector = self.SpeakerDetector(video_path, fps)
        
        # Process frames
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect faces
            faces = self._detect_faces(frame)
            
            # Check for speaking activity
            is_speaking = speaker_detector.is_speaking_at_frame(frame_idx)
            
            # Calculate optimal crop position ensuring final width is even
            crop_x = self._calculate_crop_position(
                faces, is_speaking, frame_width, target_width
            )
            # Ensure crop_x is even to maintain even pixel alignment
            crop_x = crop_x - (crop_x % 2)
            
            processed_frames.append({
                'frame_idx': frame_idx,
                'faces': faces,
                'is_speaking': is_speaking,
                'crop_x': crop_x,
                'frame_width': frame_width,
                'frame_height': frame_height,
                'target_width': target_width
            })
            
            frame_idx += 1
            
        cap.release()
        return processed_frames
        
    def _detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in a frame using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                faces.append((x, y, width, height))
                
        return faces
        
    def _calculate_crop_position(
        self, 
        faces: List[Tuple[int, int, int, int]], 
        is_speaking: bool,
        frame_width: int,
        target_width: int
    ) -> int:
        """Calculate optimal crop position using sliding window technique"""
        if not faces:
            return self.position_history[-1] if self.position_history else (frame_width - target_width) // 2
            
        # First pass: calculate base scores for all faces
        face_scores = []
        for face in faces:
            x, y, w, h = face
            face_center = x + (w // 2)
            
            # Calculate crop position to center the face
            potential_crop_x = max(0, min(
                frame_width - target_width,  # Don't exceed frame bounds
                face_center - (target_width // 2)  # Center face in crop window
            ))
            
            # Adjust crop position if face would be cut off
            face_left = x
            face_right = x + w
            crop_right = potential_crop_x + target_width
            
            # If face would be cut off on either side, adjust the crop position
            if face_left < potential_crop_x:
                # Face being cut off on left side, move crop left
                potential_crop_x = max(0, face_left)
            elif face_right > crop_right:
                # Face being cut off on right side, move crop right
                potential_crop_x = min(frame_width - target_width, face_right - target_width)
            
            # Calculate visibility after adjustments
            crop_right = potential_crop_x + target_width
            visible_left = max(face_left, potential_crop_x)
            visible_right = min(face_right, crop_right)
            
            if visible_right > visible_left:
                visibility_ratio = (visible_right - visible_left) / w
                # Base score considers size and visibility
                score = (w * h) * visibility_ratio
                # Position score based on centering
                center_offset = abs(face_center - (potential_crop_x + target_width/2)) / target_width
                position_score = 1 - center_offset
                score *= position_score
                
                # Add speaking bonus
                if is_speaking:
                    score *= 4.0  # Increased speaking bonus
                
                face_scores.append({
                    'face': face,
                    'score': score,
                    'center': face_center,
                    'crop_x': potential_crop_x
                })
        
        if not face_scores:
            return self.position_history[-1] if self.position_history else (frame_width - target_width) // 2
        
        # Second pass: adjust scores based on group positioning
        for i, face_data in enumerate(face_scores):
            group_bonus = 0
            main_face_center = face_data['center']
            
            # Check if this crop position would include other faces
            for other_face_data in face_scores:
                if other_face_data != face_data:
                    other_center = other_face_data['center']
                    # Calculate if other face would be visible in this crop
                    if (abs(other_center - main_face_center) <= target_width * 0.8):
                        group_bonus += other_face_data['score'] * 0.3  # 30% bonus for including additional faces
            
            face_data['score'] += group_bonus
        
        # Select best scoring face/position
        best_face_data = max(face_scores, key=lambda x: x['score'])
        crop_x = best_face_data['crop_x']
        
        # Apply temporal smoothing
        self.position_history.append(crop_x)
        if is_speaking:
            # More aggressive smoothing for speaking faces
            alpha = 0.8  # Increased from 0.7 for faster response to speaking faces
            smoothed_x = int(alpha * crop_x + (1 - alpha) * np.mean(list(self.position_history)[:-1]))
        else:
            # Gentler smoothing for non-speaking faces
            smoothed_x = int(np.mean(self.position_history))
        
        # Ensure the smoothed position doesn't cut off the primary face
        primary_face = best_face_data['face']
        x, _, w, _ = primary_face
        face_left = x
        face_right = x + w
        
        if face_left < smoothed_x:
            smoothed_x = max(0, face_left)
        elif face_right > smoothed_x + target_width:
            smoothed_x = min(frame_width - target_width, face_right - target_width)
        
        return smoothed_x
        
    def _create_debug_video(
        self, 
        video_path: str,
        processed_frames: List[Dict],
        output_path: str
    ) -> str:
        """Create debug visualization video"""
        if not processed_frames:
            return None
            
        cap = cv2.VideoCapture(video_path)
        frame_width = processed_frames[0]['frame_width']
        frame_height = processed_frames[0]['frame_height']
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx >= len(processed_frames):
                break
                
            frame_data = processed_frames[frame_idx]
            debug_frame = self._create_debug_frame(frame, frame_data)
            writer.write(debug_frame)
            frame_idx += 1
            
        cap.release()
        writer.release()
        return output_path
        
    def _create_debug_frame(self, frame: np.ndarray, frame_data: Dict) -> np.ndarray:
        """Create a debug visualization frame"""
        debug_frame = frame.copy()
        
        # Draw faces
        for face in frame_data['faces']:
            x, y, w, h = face
            color = (0, 165, 255)  # Orange color for all boxes
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, 2)
            
            # Add "ACTIVE" text above speaking faces
            if frame_data['is_speaking']:
                text = "ACTIVE"
                # Get text size to center it above the box
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = x + (w - text_size[0]) // 2
                text_y = max(y - 10, text_size[1])  # Ensure text doesn't go above frame
                cv2.putText(debug_frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # Green text
            
        # Draw current crop region (blue)
        cv2.rectangle(
            debug_frame,
            (frame_data['crop_x'], 0),
            (frame_data['crop_x'] + frame_data['target_width'], frame_data['frame_height']),
            (255, 0, 0), 2
        )
        
        # Draw target position indicator (red line) if in transition
        if frame_data.get('transition_frames', 0) > 0:
            target_x = frame_data.get('current_target', 0)
            cv2.line(
                debug_frame,
                (target_x, 0),
                (target_x, frame_data['frame_height']),
                (0, 0, 255), 2  # Red color
            )
            # Add transition progress text
            progress = 1 - (frame_data['transition_frames'] / (frame_data['frame_height'] * 0.5))
            cv2.putText(
                debug_frame,
                f"Transition: {progress:.1%}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        
        return debug_frame
        
    def _create_final_video(
        self,
        input_path: str,
        processed_frames: List[Dict],
        output_path: str
    ) -> str:
        """Create the final cropped video using MoviePy"""
        try:
            with VideoFileClip(input_path) as video:
                # Create a dynamic cropping function
                frame_idx = [0]  # Use list to maintain state
                
                def dynamic_crop(get_frame, t):
                    frame = get_frame(t)
                    if frame_idx[0] < len(processed_frames):
                        frame_data = processed_frames[frame_idx[0]]
                        crop_x = frame_data['crop_x']
                        target_width = frame_data['target_width']
                        frame_idx[0] += 1
                        # Ensure the cropped width is even
                        if target_width % 2 != 0:
                            target_width -= 1
                        return frame[:, crop_x:crop_x + target_width]
                    return frame
                    
                # Apply dynamic cropping
                final_clip = video.fl(dynamic_crop)
                
                # Write final video ensuring codec parameters are correct
                final_clip.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    bitrate='2500k',
                    fps=video.fps,
                    preset='fast',
                    threads=4,
                    ffmpeg_params=['-pix_fmt', 'yuv420p']  # Ensure compatible pixel format
                )
                
            return output_path
            
        except Exception as e:
            print(f"Error creating final video: {str(e)}")
            return None
            
    def _process_video_with_analysis(self, video_path: str, analysis: Dict) -> List[Dict]:
        """Process video frames using pre-analysis data"""
        processed_frames = []
        
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(5))
        
        # Calculate target width for 9:16 aspect ratio
        raw_target_width = int(frame_height * 9 / 16)
        target_width = raw_target_width if raw_target_width % 2 == 0 else raw_target_width + 1
        
        # Initialize position tracking
        current_position = frame_width // 2 - target_width // 2  # Start in center
        target_position = current_position
        transition_frames_left = 0
        max_transition_frames = int(fps * 0.5)  # 0.5 second transition
        current_focus_point_idx = 0
        
        # Sort focus points by start frame
        focus_points = sorted(analysis['focus_points'], key=lambda x: x['start_frame'])
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get current face positions
            faces = self._detect_faces(frame)
            
            # Update focus point if needed
            while (current_focus_point_idx < len(focus_points) and 
                   frame_idx >= focus_points[current_focus_point_idx]['start_frame']):
                next_focus = focus_points[current_focus_point_idx]
                target_position = next_focus['target_position']
                
                # Reset transition counter based on transition type
                if next_focus['transition'] == 'cut':
                    current_position = target_position
                    transition_frames_left = 0
                else:
                    transition_frames_left = max_transition_frames
                
                current_focus_point_idx += 1
            
            # Calculate crop position with smooth transitions
            if transition_frames_left > 0:
                # Use easing function for smoother motion
                progress = 1 - (transition_frames_left / max_transition_frames)
                # Apply cubic easing
                eased_progress = progress * progress * (3 - 2 * progress)
                current_position = int(
                    current_position + (target_position - current_position) * eased_progress
                )
                transition_frames_left -= 1
            
            # Ensure crop position is valid and even
            crop_x = max(0, min(frame_width - target_width, current_position))
            crop_x = crop_x - (crop_x % 2)
            
            # Get speaking status
            is_speaking = frame_idx < len(analysis['speaking_frames']) and analysis['speaking_frames'][frame_idx]
            
            processed_frames.append({
                'frame_idx': frame_idx,
                'faces': faces,
                'is_speaking': is_speaking,
                'crop_x': crop_x,
                'frame_width': frame_width,
                'frame_height': frame_height,
                'target_width': target_width,
                'current_target': target_position,  # Add for debugging
                'transition_frames': transition_frames_left  # Add for debugging
            })
            
            frame_idx += 1
        
        cap.release()
        return processed_frames

    class SpeakerDetector:
        """Helper class for detecting speaking activity in videos"""
        
        def __init__(self, video_path: str, fps: int):
            self.fps = fps
            self.speaking_frames = self._analyze_audio(video_path)
            
        def get_speaking_frames(self) -> List[bool]:
            """Get the list of speaking frame indicators"""
            return self.speaking_frames

        def _analyze_audio(self, video_path: str) -> List[bool]:
            """Analyze audio to detect speaking frames using enhanced detection"""
            try:
                with VideoFileClip(video_path) as video:
                    audio = video.audio
                    if (audio is None) or (audio.duration == 0):
                        print("No audio stream found in clip")
                        return [False]
                    
                    try:
                        # Create a temporary WAV file
                        temp_audio_path = os.path.splitext(video_path)[0] + "_temp_audio.wav"
                        audio.write_audiofile(temp_audio_path, fps=16000)
                        
                        # Load audio using librosa directly
                        try:
                            samples, sr = librosa.load(temp_audio_path, sr=16000, mono=True)
                            
                            # Clean up temp file
                            if os.path.exists(temp_audio_path):
                                os.remove(temp_audio_path)
                                
                        except Exception as e:
                            print(f"Error loading audio with librosa: {str(e)}")
                            return [False]
                            
                    except Exception as e:
                        print(f"Error extracting audio to WAV: {str(e)}")
                        return [False]
                    
                # Calculate frame-level features with overlap for better precision
                frame_length = int(16000 / self.fps)  # samples per video frame
                hop_length = frame_length // 2  # 50% overlap for more precise detection
                energies = []
                zero_crossings = []
                spectral_centroids = []
                
                # Process audio in frames with overlap
                for i in range(0, len(samples) - frame_length, hop_length):
                    frame = samples[i:i + frame_length]
                    if len(frame) == frame_length:
                        # Calculate energy (RMS)
                        energy = np.sqrt(np.mean(frame ** 2))
                        energies.append(energy)
                        
                        # Calculate zero-crossing rate
                        zero_crossing = np.sum(np.abs(np.diff(np.signbit(frame)))) / (2 * len(frame))
                        zero_crossings.append(zero_crossing)
                        
                        # Calculate spectral centroid for better speech/noise discrimination
                        if len(frame) >= 512:  # Ensure enough samples for FFT
                            spec = np.abs(np.fft.rfft(frame))
                            freqs = np.fft.rfftfreq(len(frame), 1/sr)
                            centroid = np.sum(freqs * spec) / (np.sum(spec) + 1e-8)
                            spectral_centroids.append(centroid)
                        else:
                            spectral_centroids.append(0)
                
                if not energies:
                    print("No valid audio frames processed")
                    return [False]
                
                # Normalize features
                energies = np.array(energies)
                zero_crossings = np.array(zero_crossings)
                spectral_centroids = np.array(spectral_centroids)
                
                # Dynamic thresholding using percentiles
                energy_threshold = np.percentile(energies, 60)  # Slightly more sensitive
                zcr_threshold = np.percentile(zero_crossings, 65)
                centroid_threshold = np.percentile(spectral_centroids, 70)
                
                # Combine features for speaking detection with weights
                speaking_frames_detailed = []
                for e, z, c in zip(energies, zero_crossings, spectral_centroids):
                    # Multi-feature speech detection with weighted scoring
                    energy_score = (e > energy_threshold) * 0.5
                    zcr_score = (z > zcr_threshold) * 0.3
                    centroid_score = (c > centroid_threshold) * 0.2
                    
                    total_score = energy_score + zcr_score + centroid_score
                    is_speaking = total_score > 0.4  # Threshold for combined features
                    speaking_frames_detailed.append(is_speaking)
                
                # Interpolate back to video frame rate (due to overlap)
                speaking_frames = []
                step = 2  # Because of 50% overlap
                for i in range(0, len(speaking_frames_detailed), step):
                    # Take maximum value in window to favor positive detections
                    window = speaking_frames_detailed[i:i + step]
                    speaking_frames.append(any(window))
                
                # Apply enhanced temporal smoothing
                window = 3  # Reduced window size for faster response
                min_speech_duration = int(0.15 * self.fps)  # Minimum 150ms for speech segment
                
                # First smoothing pass - remove isolated detections
                smoothed_frames = []
                for i in range(len(speaking_frames)):
                    start = max(0, i - window)
                    end = min(len(speaking_frames), i + window + 1)
                    window_frames = speaking_frames[start:end]
                    # Use weighted average with center bias
                    weights = np.hamming(end - start)
                    is_speaking = np.sum(np.array(window_frames) * weights) > np.sum(weights) * 0.5
                    smoothed_frames.append(is_speaking)
                
                # Second pass - enforce minimum duration
                final_frames = []
                current_segment = []
                
                for is_speaking in smoothed_frames:
                    if is_speaking:
                        current_segment.append(True)
                    else:
                        if len(current_segment) > 0:
                            # If segment is too short, mark as not speaking
                            if len(current_segment) < min_speech_duration:
                                final_frames.extend([False] * len(current_segment))
                            else:
                                final_frames.extend(current_segment)
                            current_segment = []
                        final_frames.append(False)
                
                # Handle last segment
                if current_segment:
                    if len(current_segment) < min_speech_duration:
                        final_frames.extend([False] * len(current_segment))
                    else:
                        final_frames.extend(current_segment)
                
                return final_frames
                
            except Exception as e:
                print(f"Fatal error in audio analysis: {str(e)}")
                import traceback
                traceback.print_exc()
                return [False]
                
        def is_speaking_at_frame(self, frame_idx: int) -> bool:
            """Check if there is speaking activity at a given frame"""
            if not self.speaking_frames or frame_idx >= len(self.speaking_frames):
                return False
            return self.speaking_frames[frame_idx]

class SpeechAndFaceAnalyzer:
    """Analyzes speech patterns and face positions to determine optimal camera movements"""
    
    def __init__(self):
        self.MIN_SPEECH_DURATION = 0.5  # Minimum duration in seconds to consider switching focus
        self.BRIEF_SPEECH_THRESHOLD = 1.0  # Duration threshold for brief speech
        self.POSITION_CHANGE_THRESHOLD = 0.4  # Relative frame width threshold for position change
    
    def analyze_video(self, video_path: str, fps: int) -> Dict:
        """Pre-analyze entire video for speech and face positions"""
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(3))
        total_frames = int(cap.get(7))
        print(f"Analyzing video with {total_frames} frames...")
        
        # Initialize face detection
        face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.7
        )
        
        # Detect speech for entire video first
        try:
            print("Starting audio analysis...")
            speaker_detector = VideoProcessor.SpeakerDetector(video_path, fps)
            speaking_frames = speaker_detector.get_speaking_frames()
            if not speaking_frames:
                print("Warning: No speaking frames detected, using motion-based detection...")
                # Initialize motion-based detection as fallback
                speaking_frames = [False] * total_frames
            else:
                print(f"Successfully detected {sum(speaking_frames)} speaking frames")
        except Exception as e:
            print(f"Error in speaker detection, using motion-based fallback: {str(e)}")
            speaking_frames = [False] * total_frames
        
        # Analyze face positions throughout video
        print("Starting face detection analysis...")
        face_positions = []
        face_count = 0
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect faces
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb_frame)
            
            faces = []
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    score = detection.score[0]
                    face_count += 1
                    faces.append({
                        'bbox': (x, y, width, height),
                        'center': x + (width // 2),
                        'size': width * height,
                        'score': score
                    })
            
            face_positions.append(faces)
            frame_idx += 1
            
            # Print progress every 100 frames
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames, detected {face_count} faces so far...")
        
        cap.release()
        print(f"Face detection complete. Total faces detected: {face_count}")
        
        # Analyze speech segments
        print("Analyzing speech segments...")
        speech_segments = self._find_speech_segments(speaking_frames, fps)
        print(f"Found {len(speech_segments)} speech segments")
        
        # Determine optimal focus points and transitions
        print("Calculating optimal camera movements...")
        focus_points = self._calculate_focus_points(
            speech_segments,
            face_positions,
            frame_width,
            fps
        )
        print(f"Generated {len(focus_points)} focus points for camera movement")
        
        return {
            'speech_segments': speech_segments,
            'face_positions': face_positions,
            'focus_points': focus_points,
            'total_frames': total_frames,
            'speaking_frames': speaking_frames
        }
    
    def _find_speech_segments(self, speaking_frames: List[bool], fps: int) -> List[Dict]:
        """Find continuous segments of speech"""
        segments = []
        current_segment = None
        
        for frame_idx, is_speaking in enumerate(speaking_frames):
            frame_time = frame_idx / fps
            
            if is_speaking and current_segment is None:
                # Start new segment
                current_segment = {
                    'start_frame': frame_idx,
                    'start_time': frame_time,
                    'end_frame': frame_idx,
                    'end_time': frame_time
                }
            elif is_speaking and current_segment is not None:
                # Extend current segment
                current_segment['end_frame'] = frame_idx
                current_segment['end_time'] = frame_time
            elif not is_speaking and current_segment is not None:
                # End current segment
                current_segment['duration'] = (
                    current_segment['end_time'] - current_segment['start_time']
                )
                segments.append(current_segment)
                current_segment = None
        
        # Handle last segment if it exists
        if current_segment is not None:
            current_segment['duration'] = (
                current_segment['end_time'] - current_segment['start_time']
            )
            segments.append(current_segment)
        
        return segments
    
    def _calculate_focus_points(
        self,
        speech_segments: List[Dict],
        face_positions: List[List[Dict]],
        frame_width: int,
        fps: int
    ) -> List[Dict]:
        """Calculate optimal focus points and transitions"""
        focus_points = []
        last_major_speaker_pos = None
        last_focus_point = None
        self.MIN_SPEECH_DURATION = 0.3  # Reduced from 0.5 to be more responsive
        self.BRIEF_SPEECH_THRESHOLD = 0.8  # Reduced from 1.0 to catch more speaking segments
        self.POSITION_CHANGE_THRESHOLD = 0.25  # Reduced from 0.4 to be more sensitive to movement

        # Initialize with a center focus point if we have any face data
        initial_focus = {
            'start_frame': 0,
            'target_position': frame_width // 2,
            'transition': 'smooth',
            'is_major_speaker': False
        }
        focus_points.append(initial_focus)
        last_focus_point = initial_focus  # Set initial focus point
        
        for segment in speech_segments:
            # Find the most prominent speaking face during this segment
            avg_face_pos = self._find_prominent_face_in_segment(
                segment,
                face_positions,
                frame_width
            )
            
            if avg_face_pos is None:
                continue
            
            # Determine if this is a brief interjection
            is_brief = segment['duration'] < self.BRIEF_SPEECH_THRESHOLD
            
            # Determine if position change is significant
            significant_change = True
            if last_major_speaker_pos is not None:
                rel_pos_change = abs(avg_face_pos - last_major_speaker_pos) / frame_width
                significant_change = rel_pos_change > self.POSITION_CHANGE_THRESHOLD
            
            # Decision logic for creating new focus points
            create_new_point = (
                significant_change or
                (not is_brief and segment['duration'] > self.MIN_SPEECH_DURATION)
            )
            
            if create_new_point:
                # Determine transition type based on position change
                transition_type = 'smooth'  # Default to smooth transition
                if significant_change and last_focus_point is not None:
                    # Only check for hard cuts if we have a previous focus point
                    pos_change = abs(avg_face_pos - last_focus_point['target_position']) / frame_width
                    if pos_change > 0.4:
                        transition_type = 'cut'  # Use hard cut for large position changes
                
                focus_point = {
                    'start_frame': segment['start_frame'],
                    'target_position': avg_face_pos,
                    'transition': transition_type,
                    'is_major_speaker': not is_brief,
                    'duration': segment['duration']  # Add duration for debugging
                }
                
                # Add intermediate focus point for long segments to maintain interest
                if segment['duration'] > 2.0 and len(faces_in_segment := self._get_faces_in_frame_range(
                    segment['start_frame'], 
                    segment['end_frame'],
                    face_positions
                )) > 1:
                    mid_frame = (segment['start_frame'] + segment['end_frame']) // 2
                    mid_pos = self._find_alternate_face_position(
                        avg_face_pos,
                        faces_in_segment,
                        frame_width
                    )
                    if mid_pos is not None and abs(mid_pos - avg_face_pos) / frame_width > 0.2:
                        focus_points.append({
                            'start_frame': mid_frame,
                            'target_position': mid_pos,
                            'transition': 'smooth',
                            'is_major_speaker': False,
                            'duration': 1.0  # Short duration for variety
                        })
                
                focus_points.append(focus_point)
                last_focus_point = focus_point
                
                if not is_brief:
                    last_major_speaker_pos = avg_face_pos
        
        return focus_points

    def _get_faces_in_frame_range(
        self,
        start_frame: int,
        end_frame: int,
        face_positions: List[List[Dict]]
    ) -> List[Dict]:
        """Get unique faces in a frame range"""
        unique_positions = set()
        faces = []
        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx >= len(face_positions):
                break
            for face in face_positions[frame_idx]:
                pos = face['center']
                if pos not in unique_positions:
                    unique_positions.add(pos)
                    faces.append(face)
        return faces
        
    def _find_alternate_face_position(
        self,
        current_pos: int,
        faces: List[Dict],
        frame_width: int
    ) -> Optional[int]:
        """Find an alternative face position for variety in long segments"""
        if not faces:
            return None
            
        # Filter faces that are far enough from current position
        alternate_faces = [
            face for face in faces
            if abs(face['center'] - current_pos) / frame_width > 0.2
        ]
        
        if not alternate_faces:
            return None
            
        # Choose the face with the highest score
        best_alt_face = max(alternate_faces, key=lambda x: x['score'])
        return best_alt_face['center']

    def _find_prominent_face_in_segment(
        self,
        segment: Dict,
        face_positions: List[List[Dict]],
        frame_width: int
    ) -> Optional[int]:
        """Find the most prominent face position during a speech segment"""
        if not face_positions:
            return None
            
        # Get all faces in the segment timeframe
        start_frame = segment['start_frame']
        end_frame = segment['end_frame']
        
        # Collect face data for the segment
        face_data = []
        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx >= len(face_positions):
                break
            for face in face_positions[frame_idx]:
                face_data.append({
                    'center': face['center'],
                    'size': face['size'],
                    'score': face['score']
                })
        
        if not face_data:
            return None
            
        # Group faces by similar positions (within 5% of frame width)
        position_groups = {}
        tolerance = frame_width * 0.05
        
        for face in face_data:
            center = face['center']
            matched = False
            
            for group_center in list(position_groups.keys()):
                if abs(center - group_center) < tolerance:
                    position_groups[group_center]['count'] += 1
                    position_groups[group_center]['total_score'] += face['score']
                    position_groups[group_center]['total_size'] += face['size']
                    position_groups[group_center]['centers'].append(center)
                    matched = True
                    break
            
            if not matched:
                position_groups[center] = {
                    'count': 1,
                    'total_score': face['score'],
                    'total_size': face['size'],
                    'centers': [center]
                }
        
        # Find the most prominent group
        best_group = None
        best_score = -1
        
        for center, data in position_groups.items():
            # Calculate weighted score based on frequency, detection confidence, and face size
            frequency_weight = data['count'] / len(face_data)
            avg_score = data['total_score'] / data['count']
            avg_size = data['total_size'] / data['count']
            
            # Normalize size score relative to frame width
            size_score = avg_size / (frame_width * frame_width)
            
            # Combined score with weights
            combined_score = (
                frequency_weight * 0.5 +  # 50% weight on frequency
                avg_score * 0.3 +        # 30% weight on detection confidence
                size_score * 0.2         # 20% weight on face size
            )
            
            if combined_score > best_score:
                best_score = combined_score
                best_group = data
        
        if best_group is None:
            return None
            
        # Return the median position of the best group
        return int(np.median(best_group['centers']))
        
# Create a singleton instance
video_processor = VideoProcessor()

def detect_face_and_crop(video_path: str, start: float, end: float) -> str:
    """Wrapper function for the video processor"""
    return video_processor.detect_face_and_crop(video_path, start, end)
