import cv2
import numpy as np, hashlib
import torch
import librosa
import os
import webrtcvad
import struct
import mediapipe as mp
import traceback
from moviepy.editor import VideoFileClip, AudioFileClip, VideoClip, clips_array, vfx
from typing import Tuple, List, Dict, Optional
from collections import deque
from contextlib import contextmanager
from scipy import ndimage, signal

def create_face_dict(bbox: Tuple[int, int, int, int], score: float = 1.0, face_id: int = -1) -> Dict:
    """Helper function to create a consistently structured face dictionary"""
    x, y, w, h = bbox
    return {
        'bbox': (x, y, w, h),
        'center': x + (w // 2),
        'size': w * h,
        'score': float(score),
        'id': face_id
    }

class SpeakerDetector:
    """Helper class for detecting speaking activity in videos with improved accuracy"""
    
    def __init__(self, video_path: str, fps: int):
        self.fps = fps
        self.speaking_frames = self._analyze_audio(video_path)
        
    def get_speaking_frames(self) -> List[bool]:
        """Get the list of speaking frame indicators"""
        return self.speaking_frames

    def _analyze_audio_with_vad(self, samples: np.ndarray, sr: int) -> List[float]:
        """Analyze audio using WebRTC VAD for more accurate speech detection"""
        import webrtcvad
        from array import array
        import struct
        
        # Initialize VAD with aggressive mode
        vad = webrtcvad.Vad(3)  # Aggressiveness from 0 to 3
        
        # Convert to 16-bit PCM
        if samples.dtype != np.int16:
            samples = (samples * 32768).astype(np.int16)
        
        # Process in 30ms frames (required by WebRTC VAD)
        frame_duration = 30  # ms
        samples_per_frame = int(sr * frame_duration / 1000)
        video_frame_duration = 1.0 / self.fps  # duration of one video frame
        vad_frames_per_video_frame = int(video_frame_duration * 1000 / frame_duration)  # number of VAD frames per video frame
        
        vad_scores = []
        raw_scores = []
        
        # Process VAD frames
        for i in range(0, len(samples) - samples_per_frame, samples_per_frame):
            frame = samples[i:i + samples_per_frame]
            if len(frame) == samples_per_frame:
                frame_bytes = struct.pack("%dh" % len(frame), *frame)
                try:
                    is_speech = float(vad.is_speech(frame_bytes, sr))
                    raw_scores.append(is_speech)
                except Exception as e:
                    print(f"VAD error on frame: {str(e)}")
                    raw_scores.append(0.0)
            else:
                raw_scores.append(0.0)
        
        # Aggregate VAD frames to video frames
        for i in range(0, len(raw_scores), vad_frames_per_video_frame):
            frame_scores = raw_scores[i:i + vad_frames_per_video_frame]
            if frame_scores:
                # Use weighted average with more weight on recent frames
                weights = np.linspace(0.5, 1.0, len(frame_scores))
                weighted_score = np.average(frame_scores, weights=weights)
                vad_scores.append(weighted_score)
            else:
                vad_scores.append(0.0)
        
        # Apply temporal smoothing
        window_size = 5
        smoothed_scores = []
        
        for i in range(len(vad_scores)):
            start = max(0, i - window_size // 2)
            end = min(len(vad_scores), i + window_size // 2 + 1)
            window = vad_scores[start:end]
            # Use exponential moving average for smoothing
            alpha = 0.7
            if i > 0:
                smoothed_score = alpha * np.mean(window) + (1 - alpha) * smoothed_scores[-1]
            else:
                smoothed_score = np.mean(window)
            smoothed_scores.append(smoothed_score)
        
        print(f"Generated {len(smoothed_scores)} VAD scores for {len(samples)/sr:.2f}s of audio")
        if len(smoothed_scores) > 0:
            print(f"Score range: {min(smoothed_scores):.3f} - {max(smoothed_scores):.3f}")
            print(f"Sample scores: {smoothed_scores[:10]}")
        
        return smoothed_scores

    def _analyze_audio(self, video_path: str) -> List[float]:
        """Analyze audio to detect speaking frames using enhanced VAD and audio features"""
        try:
            print(f"Starting enhanced audio analysis for: {video_path}")
            
            # Load audio using MoviePy first to ensure we have valid audio
            with VideoFileClip(video_path) as video:
                if video.audio is None:
                    print("Warning: Video has no audio stream")
                    return [0.0]
                
                # Create temporary WAV file with specific parameters for VAD
                temp_audio_path = os.path.splitext(video_path)[0] + "_temp_audio.wav"
                try:
                    video.audio.write_audiofile(temp_audio_path, fps=16000, nbytes=2, 
                                             codec='pcm_s16le', ffmpeg_params=["-ac", "1"])
                    
                    print("Loading audio with librosa...")
                    samples, sr = librosa.load(temp_audio_path, sr=16000, mono=True)
                    if len(samples) == 0:
                        print("Error: No audio samples loaded")
                        return [0.0]
                    
                    print(f"Audio loaded: {len(samples)} samples at {sr}Hz")
                    
                    # Calculate frame-level features
                    frame_length = int(16000 / self.fps)  # Audio samples per video frame
                    hop_length = frame_length // 4
                    
                    print("Computing audio features...")
                    
                    # 1. Get VAD scores
                    vad_scores = self._analyze_audio_with_vad(samples, sr)
                    if not vad_scores:
                        print("VAD analysis failed")
                        return [0.0]
                    
                    # 2. Compute RMS energy
                    rms = librosa.feature.rms(y=samples, frame_length=frame_length, 
                                            hop_length=hop_length)[0]
                    rms = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)
                    
                    # 3. Compute spectral centroid (indicates voice presence)
                    cent = librosa.feature.spectral_centroid(y=samples, sr=sr,
                                                           n_fft=frame_length,
                                                           hop_length=hop_length)[0]
                    cent = (cent - cent.min()) / (cent.max() - cent.min() + 1e-8)
                    
                    # 4. Zero-crossing rate (helps distinguish voiced/unvoiced)
                    zcr = librosa.feature.zero_crossing_rate(y=samples,
                                                           frame_length=frame_length,
                                                           hop_length=hop_length)[0]
                    zcr = (zcr - zcr.min()) / (zcr.max() - zcr.min() + 1e-8)
                    
                    print("Combining features...")
                    
                    # Combine all features with weighted importance
                    combined_scores = []
                    min_len = min(len(vad_scores), len(rms), len(cent), len(zcr))
                    
                    for i in range(min_len):
                        # Weight VAD most heavily but consider other features
                        score = (vad_scores[i] * 0.6 +    # VAD gets highest weight
                                rms[i] * 0.2 +            # Energy
                                cent[i] * 0.1 +           # Spectral centroid
                                zcr[i] * 0.1)             # Zero-crossing rate
                        combined_scores.append(score)
                    
                    # Apply temporal smoothing
                    window_size = 5
                    smoothed_scores = []
                    
                    for i in range(len(combined_scores)):
                        start_idx = max(0, i - window_size // 2)
                        end_idx = min(len(combined_scores), i + window_size // 2 + 1)
                        window = combined_scores[start_idx:end_idx]
                        smoothed_scores.append(sum(window) / len(window))
                    
                    print(f"Analysis complete. Generated {len(smoothed_scores)} frame scores")
                    
                    # Clean up
                    if os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
                        print("Cleaned up temporary audio file")
                    
                    return smoothed_scores
                    
                except Exception as e:
                    print(f"Error in audio analysis: {str(e)}")
                    if os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
                    return [0.0]
                    
        except Exception as e:
            print(f"Fatal error in audio analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return [0.0]

    def is_speaking_at_frame(self, frame_idx: int) -> bool:
        """Check if there is speaking activity at a given frame with improved thresholding"""
        if not self.speaking_frames or frame_idx >= len(self.speaking_frames):
            return False
        
        # Use dynamic thresholding based on local context
        window_size = 5
        start_idx = max(0, frame_idx - window_size // 2)
        end_idx = min(len(self.speaking_frames), frame_idx + window_size // 2 + 1)
        window = self.speaking_frames[start_idx:end_idx]
        
        if not window:
            return False
        
        current_score = self.speaking_frames[frame_idx]
        local_mean = sum(window) / len(window)
        
        # Adaptive threshold: lower threshold if local activity is high
        base_threshold = 0.4
        if local_mean > 0.3:  # If there's significant activity in the window
            threshold = base_threshold * 0.8  # Lower threshold by 20%
        else:
            threshold = base_threshold
        
        return current_score > threshold

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
        print(f"\nAnalyzing video with {total_frames} frames...")
        
        # Initialize face detection with lower confidence threshold
        face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # 0 for short-range, 1 for full-range
            min_detection_confidence=0.5  # Lowered from 0.7 for better detection
        )
        
        # Detect speech for entire video first
        try:
            print("\nStarting audio analysis...")
            speaker_detector = SpeakerDetector(video_path, fps)  
            speaking_frames = speaker_detector.get_speaking_frames()
            if not speaking_frames:
                print("Warning: No speaking frames detected!")
                speaking_frames = [False] * total_frames
            else:
                speaking_count = sum(1 for score in speaking_frames if score > 0.4)
                print(f"Successfully detected {speaking_count} speaking frames out of {len(speaking_frames)}")
                print(f"Sample of speech scores: {speaking_frames[:10]}")
        except Exception as e:
            print(f"Error in speaker detection: {str(e)}")
            import traceback
            traceback.print_exc()
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
                h, w, _ = frame.shape
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    faces.append(create_face_dict(bbox=(x, y, width, height), score=float(detection.score[0])))
                    face_count += 1
            
            face_positions.append(faces)
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames, detected {face_count} faces so far...")
        
        cap.release()
        print(f"Face detection complete. Total faces detected: {face_count}")
        
        # Track faces to maintain consistent IDs
        print("Tracking faces across frames...")
        face_positions = self._track_faces(face_positions, frame_width)
        
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
        
        # Stricter thresholds for more accurate speaker detection
        self.MIN_SPEECH_DURATION = 0.75  # increased to reduce false positives
        self.BRIEF_SPEECH_THRESHOLD = 1.5  # increased to better identify main speakers
        self.POSITION_CHANGE_THRESHOLD = 0.35  # slightly reduced for more responsive tracking

        # Initialize with a center focus point
        initial_focus = {
            'start_frame': 0,
            'target_position': frame_width // 2,
            'transition': 'smooth',
            'is_major_speaker': False
        }
        focus_points.append(initial_focus)
        last_focus_point = initial_focus

        for segment in speech_segments:
            # Skip very short segments (likely noise)
            if segment['duration'] < self.MIN_SPEECH_DURATION:
                continue
                
            # Find the most prominent face during this segment
            face_data = self._find_prominent_face_in_segment(
                segment,
                face_positions,
                frame_width
            )
            
            if face_data is None:
                continue
            
            # Extract the center position from the face data
            avg_face_pos = face_data['center']
            
            # Determine if this is a brief interjection
            is_brief = segment['duration'] < self.BRIEF_SPEECH_THRESHOLD
            
            # Additional validation: check face consistency
            face_consistency = self._validate_face_consistency(
                segment,
                face_positions,
                face_data,  # Pass the entire face_data dictionary
                frame_width
            )
            
            if not face_consistency and not is_brief:
                continue  # Skip if face position is not consistent during speech
            
            # Determine if position change is significant
            significant_change = True
            if last_major_speaker_pos is not None:
                rel_pos_change = abs(avg_face_pos - last_major_speaker_pos) / frame_width
                significant_change = rel_pos_change > self.POSITION_CHANGE_THRESHOLD
            
            # More stringent decision logic for creating new focus points
            create_new_point = (
                (significant_change and segment['duration'] > self.MIN_SPEECH_DURATION) or
                (not is_brief and segment['duration'] > self.BRIEF_SPEECH_THRESHOLD and face_consistency)
            )
            
            if create_new_point:
                # Improved transition type decision
                transition_type = 'smooth'
                if significant_change and last_focus_point is not None:
                    pos_change = abs(avg_face_pos - last_focus_point['target_position']) / frame_width
                    # Use hard cuts only for very significant changes during longer speech segments
                    if pos_change > 0.45 and segment['duration'] > self.BRIEF_SPEECH_THRESHOLD:
                        transition_type = 'cut'
                
                focus_point = {
                    'start_frame': segment['start_frame'],
                    'target_position': avg_face_pos,
                    'transition': transition_type,
                    'is_major_speaker': not is_brief and face_consistency,
                    'duration': segment['duration']
                }
                
                # Only add intermediate focus points for very long segments with verified multiple speakers
                if (segment['duration'] > 3.0 and  # Increased duration threshold
                    face_consistency and
                    len(faces_in_segment := self._get_faces_in_frame_range(
                        segment['start_frame'], 
                        segment['end_frame'],
                        face_positions
                    )) > 1):
                    
                    mid_frame = (segment['start_frame'] + segment['end_frame']) // 2
                    mid_pos_data = self._find_alternate_face_position(
                        avg_face_pos,
                        faces_in_segment,
                        frame_width
                    )
                    
                    if (mid_pos_data is not None and 
                        abs(mid_pos_data['center'] - avg_face_pos) / frame_width > 0.25 and  # Increased threshold
                        self._validate_face_consistency(  # Validate alternate face too
                            {'start_frame': mid_frame - fps, 'end_frame': mid_frame + fps},
                            face_positions,
                            mid_pos_data,
                            frame_width
                        )):
                        focus_points.append({
                            'start_frame': mid_frame,
                            'target_position': mid_pos_data['center'],
                            'transition': 'smooth',
                            'is_major_speaker': False,
                            'duration': 1.0
                        })
                
                focus_points.append(focus_point)
                last_focus_point = focus_point
                
                if not is_brief and face_consistency:
                    last_major_speaker_pos = avg_face_pos
        
        return focus_points

    def _validate_face_consistency(
        self,
        segment: Dict,
        face_positions: List[List[Dict]],
        face_pos: Dict,
        frame_width: int
    ) -> bool:
        """Validate that a face position is consistently present during a speech segment"""
        if face_pos is None:
            return False
            
        presence_count = 0
        total_frames = 0
        tolerance = frame_width * 0.1  # 10% of frame width tolerance
        target_center = face_pos['center']  # Extract the center position
        
        for frame_idx in range(segment['start_frame'], segment['end_frame'] + 1):
            if frame_idx >= len(face_positions):
                break
                
            total_frames += 1
            face_found = False
            
            for face in face_positions[frame_idx]:
                if abs(face['center'] - target_center) <= tolerance:
                    face_found = True
                    presence_count += 1
                    break
        
        if total_frames == 0:
            return False
            
        # Require face to be present in at least 70% of frames
        return (presence_count / total_frames) >= 0.7

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
    ) -> Optional[Dict]:
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
        return {
            'center': best_alt_face['center'],
            'likelihood_score': best_alt_face['score'],
            'presence_ratio': 1.0,  # Single frame presence
            'is_likely_speaker': False
        }

    def _find_prominent_face_in_segment(
        self,
        segment: Dict,
        face_positions: List[List[Dict]],
        frame_width: int
    ) -> Optional[Dict]:
        """Find the most likely speaking face with improved confidence metrics"""
        if not face_positions:
            return None

        start_frame = segment['start_frame']
        end_frame = segment['end_frame']
        
        # Enhanced face tracking with temporal consistency
        face_tracks = {}  # Track faces across frames
        frame_count = end_frame - start_frame + 1

        # Step 1: Track faces across frames using position and size consistency
        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx >= len(face_positions):
                break

            frame_faces = face_positions[frame_idx]
            for face in frame_faces:
                face_center = float(face['center'])
                face_size = face['size']
                
                # Find matching track or create new one
                matched_track = None
                for track_id, track_data in face_tracks.items():
                    last_pos = track_data['positions'][-1]
                    last_size = track_data['sizes'][-1]
                    
                    # Check if this face matches an existing track
                    pos_diff = abs(face_center - last_pos) / frame_width
                    size_ratio = min(face_size, last_size) / max(face_size, last_size)
                    
                    if pos_diff < 0.1 and size_ratio > 0.7:  # Adjusted thresholds
                        matched_track = track_id
                        break

                if matched_track is None:
                    # Create new track
                    track_id = len(face_tracks)
                    face_tracks[track_id] = {
                        'positions': [],
                        'sizes': [],
                        'scores': [],
                        'frame_indices': []
                    }
                    matched_track = track_id

                # Update track
                track = face_tracks[matched_track]
                track['positions'].append(face_center)
                track['sizes'].append(face_size)
                track['scores'].append(face['score'])
                track['frame_indices'].append(frame_idx)

        # Step 2: Score each face track with enhanced metrics
        track_scores = {}
        for track_id, track_data in face_tracks.items():
            # Basic presence score
            presence_ratio = len(track_data['frame_indices']) / frame_count
            
            # Position stability (lower variance is better)
            pos_variance = np.var(track_data['positions']) / frame_width if len(track_data['positions']) > 1 else 1.0
            stability_score = 1.0 / (1.0 + pos_variance)
            
            # Size consistency (prefer consistent sizes)
            size_variance = np.var(track_data['sizes']) / np.mean(track_data['sizes']) if len(track_data['sizes']) > 1 else 1.0
            size_consistency = 1.0 / (1.0 + size_variance)
            
            # Detection confidence
            avg_detection_score = np.mean(track_data['scores'])
            
            # Central tendency (prefer faces closer to center)
            avg_position = np.mean(track_data['positions'])
            centrality = 1.0 - abs(avg_position - frame_width/2) / (frame_width/2)
            
            # Temporal consistency (prefer faces that maintain consistent presence)
            frame_gaps = np.diff(track_data['frame_indices'])
            temporal_consistency = 1.0 / (1.0 + np.mean(frame_gaps)) if len(frame_gaps) > 0 else 0.0
            
            # Combine scores with weighted importance
            track_scores[track_id] = {
                'likelihood_score': (
                    presence_ratio * 0.35 +        # Most important: consistent presence
                    stability_score * 0.20 +       # Stable position
                    avg_detection_score * 0.15 +   # Reliable detection
                    size_consistency * 0.15 +      # Consistent size
                    centrality * 0.10 +            # Somewhat centered
                    temporal_consistency * 0.05     # Few frame gaps
                ),
                'center': int(np.median(track_data['positions'])),  # Use median for robustness
                'presence_ratio': presence_ratio,
                'is_likely_speaker': True  # Will be determined by threshold
            }

        if not track_scores:
            return None

        # Find best track that meets minimum threshold
        best_track = max(track_scores.items(), key=lambda x: x[1]['likelihood_score'])
        best_score = best_track[1]['likelihood_score']

        # Only return if we're confident about the speaker
        if best_score > 0.6:  # Increased confidence threshold
            return best_track[1]
        
        return None

    def _track_faces(self, face_positions: List[List[Dict]], frame_width: int) -> List[List[Dict]]:
        """Track and maintain consistent face IDs across frames"""
        tracked_faces = []
        face_history = []  # Track face positions over time
        next_face_id = 0
        position_tolerance = float(frame_width * 0.1)  # 10% of frame width

        for frame_faces in face_positions:
            current_tracked = []
            used_ids = set()

            # Try to match current faces with existing tracks
            for face in frame_faces:
                face_center = float(face['center'])
                best_match_id = None
                best_match_dist = float(position_tolerance)

                # Look back up to 15 frames to find matches
                for prev_frame in face_history[-15:]:
                    for tracked_face in prev_frame:
                        if tracked_face['id'] not in used_ids:
                            dist = abs(float(tracked_face['center']) - face_center)
                            if dist < best_match_dist:
                                best_match_dist = dist
                                best_match_id = tracked_face['id']

                if best_match_id is not None:
                    # Matched with existing track
                    face['id'] = best_match_id
                    used_ids.add(best_match_id)
                else:
                    # New face detected
                    face['id'] = next_face_id
                    next_face_id += 1

                current_tracked.append(face)

            tracked_faces.append(current_tracked)
            face_history.append(current_tracked)

            # Limit history size
            if len(face_history) > 30:  # Keep 1 second of history at 30fps
                face_history.pop(0)

        return tracked_faces

class VideoProcessor:
    """Main class for processing videos into YouTube Shorts format"""
    
    def __init__(self):
        """Initialize the video processor with aggressive face detection"""
        # Initialize MediaPipe face detection with lower confidence threshold for better detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for full-range detection
            min_detection_confidence=0.4  # Lowered even further for more aggressive detection
        )
        
        # Initialize device for processing
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize tracking parameters for aggressive movement
        self.speaking_window = None
        self.speaking_history = None
        self.tracking_window = None
        self.position_history = None
        self.active_speaker_duration = 0
        
        # Set up paths
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.temp_dir = os.path.join(self.project_dir, "temp_files")
        self.finished_dir = os.path.join(self.project_dir, "finished_videos")
        
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.finished_dir, exist_ok=True)
        
        self.analyzer = None

    def _initialize_for_video(self, video_path: str) -> int:
        """Initialize processor settings for a specific video"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
        if fps <= 0:
            print(f"Warning: Invalid fps ({fps}) detected, using default of 30")
            fps = 30
        
        # Initialize fps-dependent parameters
        self.speaking_window = int(fps * 0.5)  # 0.5 second window
        self.speaking_history = deque(maxlen=self.speaking_window)
        
        self.tracking_window = int(fps * 1.0)  # 1 second window
        self.position_history = deque(maxlen=self.tracking_window)
        
        # Initialize analyzer with correct fps
        self.analyzer = SpeechAndFaceAnalyzer()
        
        print(f"Initialized processor for video with {fps} fps")
        print(f"Speaking window: {self.speaking_window} frames")
        print(f"Tracking window: {self.tracking_window} frames")
        
        return fps

    def detect_face_and_crop(self, video_path: str, start: float, end: float) -> str:
        """Main entry point for processing a video segment"""
        try:
            print("\n=== Starting Video Processing Pipeline ===")
            
            # Initialize for this specific video
            fps = self._initialize_for_video(video_path)
            
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
            cap = cv2.VideoCapture(temp_clip_path)
            fps = int(cap.get(5))  # cv2.CAP_PROP_FPS is 5
            cap.release()
            
            # Use the existing analyzer instance instead of creating a new one
            analysis = self.analyzer.analyze_video(temp_clip_path, fps)
            
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
        speaker_detector = SpeakerDetector(video_path, fps)
        
        # Initialize active speaker duration counter
        self.active_speaker_duration = 0

        # Process frames
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect faces
            faces = self._detect_faces(frame)
            
            # Check for speaking activity at current frame
            is_speaking = speaker_detector.is_speaking_at_frame(frame_idx)
            
            # Update speaking history and determine active speaker over last 5 frames
            self.speaking_history.append(is_speaking)
            active_speaker = False
            if len(self.speaking_history) >= 5:
                # If at least 3 out of the last 5 frames had speaking activity, consider as active speaker
                if sum(list(self.speaking_history)[-5:]) >= 3:
                    active_speaker = True
            
            # Update active speaker duration counter
            if active_speaker:
                self.active_speaker_duration += 1
            else:
                self.active_speaker_duration = 0
            
            # Calculate optimal crop position ensuring final width is even
            crop_x = self._calculate_crop_position(
                faces, is_speaking, frame_width, target_width
            )
            crop_x = crop_x - (crop_x % 2)  # Ensure even alignment
            
            processed_frames.append({
                'frame_idx': frame_idx,
                'faces': faces,
                'is_speaking': is_speaking,
                'active_speaker': active_speaker,  # New field for active speaker status
                'crop_x': crop_x,
                'frame_width': frame_width,
                'frame_height': frame_height,
                'target_width': target_width
            })
            
            frame_idx += 1
            
        cap.release()
        return processed_frames
        
    def _detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces in a frame using MediaPipe and return consistent dictionary format"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)
        
        faces = []
        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Calculate center position
                center = x + (width // 2)
                
                # Create face dictionary with all required fields
                face_dict = {
                    'bbox': (x, y, width, height),
                    'center': center,
                    'size': width * height,
                    'score': float(detection.score[0]),
                    'id': -1  # Default ID for untracked faces
                }
                faces.append(face_dict)
                
        return faces
        
    def _calculate_crop_position(
        self, 
        faces: List[Tuple[int, int, int, int]], 
        is_speaking: bool,
        frame_width: int,
        target_width: int
    ) -> int:
        """Calculate optimal crop position using sliding window technique with improved active speaker handling
        to ensure the full face plus a buffer is included in the crop window and remains centered when possible"""
        if not faces:
            return self.position_history[-1] if self.position_history else (frame_width - target_width) // 2
            
        # First pass: calculate base scores for all faces
        face_scores = []
        for face in faces:
            x, y, w, h = face
            face_center = x + (w // 2)
            
            # Calculate initial crop position to center the face
            base_crop_x = face_center - target_width // 2
            
            # Compute a buffer based on face width
            buffer = max(2, int(0.1 * w))
            
            # Determine allowed crop range to fully include the face plus buffer
            face_left = x
            face_right = x + w
            lower_bound = max(0, face_right + buffer - target_width)
            upper_bound = min(frame_width - target_width, face_left - buffer)
            if lower_bound <= upper_bound:
                potential_crop_x = max(lower_bound, min(base_crop_x, upper_bound))
            else:
                potential_crop_x = base_crop_x
            
            # Calculate visibility after adjustments
            crop_right = potential_crop_x + target_width
            visible_left = max(face_left, potential_crop_x)
            visible_right = min(face_right, crop_right)
            
            if visible_right > visible_left:
                visibility_ratio = (visible_right - visible_left) / w
                score = (w * h) * visibility_ratio
                center_offset = abs(face_center - (potential_crop_x + target_width/2)) / target_width
                position_score = 1 - center_offset
                score *= position_score
                if is_speaking:
                    score *= 4.0
                face_scores.append({
                    'face': face,
                    'score': score,
                    'center': face_center,
                    'crop_x': potential_crop_x
                })
        
        if not face_scores:
            return self.position_history[-1] if self.position_history else (frame_width - target_width) // 2
        
        # Second pass: adjust scores based on group positioning
        for face_data in face_scores:
            group_bonus = 0
            main_face_center = face_data['center']
            for other_face_data in face_scores:
                if other_face_data != face_data:
                    other_center = other_face_data['center']
                    if abs(other_center - main_face_center) <= target_width * 0.8:
                        group_bonus += other_face_data['score'] * 0.3
            face_data['score'] += group_bonus
        
        best_face_data = max(face_scores, key=lambda x: x['score'])
        candidate_crop_x = best_face_data['crop_x']
        active_face_center = best_face_data['center']
        
        # Determine current crop position from history if available
        if self.position_history:
            current_crop_x = self.position_history[-1]
        else:
            current_crop_x = candidate_crop_x
        
        # Recalculate buffer for the active face
        x, y, w, h = best_face_data['face']
        buffer = max(2, int(0.1 * w))
        desired_left = max(0, x - buffer)
        desired_right = min(frame_width, x + w + buffer)

        # If the active face (with buffer) is fully within the current crop window, remain still
        if current_crop_x <= desired_left and (current_crop_x + target_width) >= desired_right:
            new_crop_x = current_crop_x
        else:
            # Calculate distance from center of current window to active face
            current_center = current_crop_x + target_width / 2
            diff = abs(active_face_center - current_center)
            if diff > target_width / 2:
                # Use jump cut only if active speaker has been speaking long enough
                if self.active_speaker_duration >= 10:
                    new_crop_x = candidate_crop_x
                else:
                    # Otherwise, use a smoother tween
                    alpha = 0.5
                    new_crop_x = int(alpha * candidate_crop_x + (1 - alpha) * current_crop_x)
            else:
                # Smooth tween: transition gradually
                alpha = 0.8 if is_speaking else 0.5
                new_crop_x = int(alpha * candidate_crop_x + (1 - alpha) * current_crop_x)
        
        # Ensure new_crop_x is even for alignment
        new_crop_x = new_crop_x - (new_crop_x % 2)
        
        # Final check: adjust to ensure the active face is fully within the crop
        if x < new_crop_x:
            new_crop_x = max(0, x)
        elif (x + w) > (new_crop_x + target_width):
            new_crop_x = min(frame_width - target_width, x + w - target_width)
        
        # Update position history
        self.position_history.append(new_crop_x)
        
        return new_crop_x
        
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
        # Convert dimensions to integers for VideoWriter
        frame_width = int(processed_frames[0]['frame_width'])
        frame_height = int(processed_frames[0]['frame_height'])
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create VideoWriter with integer dimensions
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
            # Ensure integer dimensions in frame_data for visualization
            frame_data = {
                **frame_data,
                'frame_width': int(frame_data['frame_width']),
                'frame_height': int(frame_data['frame_height']),
                'target_width': int(frame_data['target_width']),
                'crop_x': int(frame_data['crop_x'])
            }
            debug_frame = self._create_debug_frame(frame, frame_data)
            writer.write(debug_frame)
            frame_idx += 1
            
        cap.release()
        writer.release()
        return output_path
        
    def _create_debug_frame(self, frame: np.ndarray, frame_data: Dict) -> np.ndarray:
        """Create debug visualization frame with better visualization of tracking"""
        debug_frame = frame.copy()
        
        # Define colors in BGR format for OpenCV
        COLORS = {
            'text_bg': (0, 0, 0),        # Black
            'active': (0, 255, 0),       # Green
            'inactive': (0, 165, 255),    # Orange
            'crop': (255, 0, 0),         # Blue
            'metrics': (255, 255, 0)      # Yellow
        }

        # Draw metrics panel with transparency
        overlay = debug_frame.copy()
        metrics_height = 120
        cv2.rectangle(overlay, (0, 0), (int(frame_data['frame_width']), metrics_height), 
                     COLORS['text_bg'], -1)
        alpha = 0.7
        debug_frame = cv2.addWeighted(overlay, alpha, debug_frame, 1 - alpha, 0)

        # Display metrics
        metrics_text = [
            f"Frame: {frame_data['frame_idx']}",
            f"Speaking: {'YES' if frame_data.get('is_speaking') else 'NO'}",
            f"Crop Position: {frame_data['crop_x']:.1f}",
            f"Target Width: {frame_data['target_width']:.1f}",
            f"Face Count: {len(frame_data['faces'])}"
        ]
        
        y_offset = 30
        for text in metrics_text:
            cv2.putText(debug_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       COLORS['metrics'], 2)
            y_offset += 20

        # Draw detected faces with centers
        if frame_data.get('faces'):
            for face in frame_data['faces']:
                x, y, w, h = face['bbox']
                center = face['center']
                score = face['score']
                
                # Draw face rectangle
                color = COLORS['active'] if frame_data.get('is_speaking') else COLORS['inactive']
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw face center
                cv2.circle(debug_frame, (int(center), y + h//2), 4, color, -1)
                
                # Draw face info
                info_text = f"Score: {score:.2f}"
                cv2.putText(debug_frame, info_text, (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw crop region more visibly
        crop_x = int(frame_data['crop_x'])
        target_width = int(frame_data['target_width'])
        frame_height = int(frame_data['frame_height'])
        
        # Draw vertical lines for crop region
        cv2.line(debug_frame, (crop_x, 0), (crop_x, frame_height), COLORS['crop'], 2)
        cv2.line(debug_frame, (crop_x + target_width, 0), (crop_x + target_width, frame_height), COLORS['crop'], 2)
        
        # Draw horizontal lines to complete the box
        cv2.line(debug_frame, (crop_x, 0), (crop_x + target_width, 0), COLORS['crop'], 2)
        cv2.line(debug_frame, (crop_x, frame_height - 1), (crop_x + target_width, frame_height - 1), COLORS['crop'], 2)
        
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
                        # Convert dimensions to integers and ensure they're valid
                        crop_x = int(frame_data['crop_x'])
                        target_width = int(frame_data['target_width'])
                        
                        # Ensure crop region is within frame bounds
                        frame_width = frame.shape[1]
                        crop_x = max(0, min(frame_width - target_width, crop_x))
                        
                        # Crop the frame
                        cropped = frame[:, crop_x:crop_x + target_width]
                        frame_idx[0] += 1
                        return cropped
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
            traceback.print_exc()
            return None
            
    def _process_video_with_analysis(self, video_path: str, analysis: Dict) -> List[Dict]:
        """Process video frames using pre-analysis data with aggressive face tracking"""
        processed_frames = []
        
        cap = cv2.VideoCapture(video_path)
        frame_width = float(cap.get(3))
        frame_height = float(cap.get(4))
        fps = float(cap.get(5))
        
        # Calculate target width for 9:16 aspect ratio
        target_width = float(int(frame_height * 9 / 16))
        if target_width % 2 != 0:
            target_width += 1.0
        
        # Initialize with centered position
        current_position = float((frame_width - target_width) / 2)
        target_position = current_position
        
        # Tracking state
        current_tracked_face = None
        stable_position_count = 0
        STABILITY_THRESHOLD = 8  # Increased frames needed for stability
        MOVEMENT_THRESHOLD = target_width * 0.03  # Reduced to 3% for less jitter
        LOCK_THRESHOLD = 15  # Frames needed to lock position
        TRANSITION_FRAMES = 10  # Frames to smooth transition between locked positions
        VELOCITY_DAMPING = 0.85  # Damping factor for velocity
        last_significant_move = 0
        position_locked = False
        locked_position = None
        transition_start_pos = None
        transition_end_pos = None
        transition_frame_count = 0
        velocity = 0.0  # Current velocity for smooth movement
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Get faces and speech info
            faces = self._detect_faces(frame) 
            
            # Get speaking state
            is_speaking = False
            if frame_idx < len(analysis['speaking_frames']):
                is_speaking = analysis['speaking_frames'][frame_idx] > 0.4
            
            # First priority: Find speaking faces
            speaking_faces = []
            non_speaking_faces = []
            
            for face in faces:
                if is_speaking:
                    speaking_faces.append(face)
                else:
                    non_speaking_faces.append(face)
            
            # Target selection with stability
            new_target = None
            
            if speaking_faces:
                # Find the most central speaking face
                best_speaking_face = min(speaking_faces, 
                                       key=lambda f: abs(f['center'] - frame_width/2))
                
                if current_tracked_face is None:
                    # No current track, start tracking this face
                    new_target = best_speaking_face['center']
                    current_tracked_face = best_speaking_face
                    stable_position_count = 0
                else:
                    # Check if we should switch to the new speaking face
                    current_center = current_tracked_face['center']
                    new_center = best_speaking_face['center']
                    distance = abs(new_center - current_center)
                    
                    # Only switch if the new face is significantly different
                    if distance > target_width * 0.3:  # 30% of target width
                        new_target = new_center
                        current_tracked_face = best_speaking_face
                        stable_position_count = 0
                        last_significant_move = frame_idx
                    else:
                        # Keep tracking current face with stability
                        new_target = current_center
                        stable_position_count += 1
            
            elif non_speaking_faces:
                if current_tracked_face is None:
                    # No current track, find most central face
                    best_face = min(non_speaking_faces, 
                                  key=lambda f: abs(f['center'] - frame_width/2))
                    new_target = best_face['center']
                    current_tracked_face = best_face
                    stable_position_count = 0
                else:
                    # Try to find current tracked face
                    found = False
                    for face in non_speaking_faces:
                        if abs(face['center'] - current_tracked_face['center']) < target_width * 0.2:
                            current_tracked_face = face
                            new_target = face['center']
                            found = True
                            stable_position_count += 1
                            break
                    
                    if not found:
                        # Only switch if we've been stable for a while
                        if frame_idx - last_significant_move > fps * 1.0:  # 1 second
                            best_face = min(non_speaking_faces, 
                                          key=lambda f: abs(f['center'] - frame_width/2))
                            new_target = best_face['center']
                            current_tracked_face = best_face
                            stable_position_count = 0
                            last_significant_move = frame_idx
            
            # Position update with velocity dampening
            if new_target is not None:
                target_center = new_target - (target_width / 2)
                
                if transition_frame_count > 0:
                    # Smooth transition with velocity dampening
                    progress = transition_frame_count / TRANSITION_FRAMES
                    target_pos = transition_start_pos + (transition_end_pos - transition_start_pos) * (1 - progress)
                    velocity = (target_pos - current_position) * 0.2
                    velocity *= VELOCITY_DAMPING
                    current_position += velocity
                    transition_frame_count -= 1
                    if transition_frame_count == 0:
                        position_locked = True
                        locked_position = transition_end_pos
                        velocity = 0.0
                elif position_locked:
                    if abs(target_center - locked_position) > target_width * 0.3:
                        # Start transition with reset velocity
                        transition_start_pos = locked_position
                        transition_end_pos = target_center
                        transition_frame_count = TRANSITION_FRAMES
                        position_locked = False
                        stable_position_count = 0
                        velocity = 0.0
                    else:
                        # Stay perfectly still when locked
                        current_position = locked_position
                        velocity = 0.0
                else:
                    if stable_position_count >= STABILITY_THRESHOLD:
                        if abs(target_center - current_position) <= MOVEMENT_THRESHOLD:
                            stable_position_count += 1
                            if stable_position_count >= LOCK_THRESHOLD:
                                position_locked = True
                                locked_position = current_position
                                velocity = 0.0
                        else:
                            # Very gentle movement when near stable
                            velocity = (target_center - current_position) * 0.05
                            velocity *= VELOCITY_DAMPING
                            current_position += velocity
                    else:
                        current_distance = abs(target_center - current_position)
                        if current_distance > target_width * 0.1:
                            if current_tracked_face and current_tracked_face.get('score', 0) > 0.8:
                                # Smooth transition for high confidence faces
                                transition_start_pos = current_position
                                transition_end_pos = target_center
                                transition_frame_count = TRANSITION_FRAMES
                                velocity = 0.0
                            else:
                                # Direct jump with velocity reset
                                current_position = target_center
                                velocity = 0.0
                            stable_position_count = 0
                        else:
                            # Normal movement with velocity dampening
                            velocity = (target_center - current_position) * 0.2
                            velocity *= VELOCITY_DAMPING
                            current_position += velocity
                
                # Ensure bounds with velocity reset at boundaries
                old_position = current_position
                current_position = float(np.clip(current_position, 0, frame_width - target_width))
                if current_position != old_position:
                    velocity = 0.0  # Reset velocity if we hit boundaries
            
            # Ensure even position for display
            display_position = int(current_position)
            if display_position % 2 != 0:
                display_position -= 1
            
            # Store frame data with lock status
            processed_frames.append({
                'frame_idx': frame_idx,
                'faces': faces,
                'is_speaking': is_speaking,
                'audio_measure': analysis['speaking_frames'][frame_idx] if frame_idx < len(analysis['speaking_frames']) else 0.0,
                'crop_x': display_position,
                'frame_width': frame_width,
                'frame_height': frame_height,
                'target_width': target_width,
                'speaker_confidence': current_tracked_face['score'] if current_tracked_face else 0.0,
                'speaker_duration': stable_position_count,
                'position_locked': position_locked
            })
            
            frame_idx += 1
        
        cap.release()
        return processed_frames

    def _get_face_color(self, face_id: int) -> Tuple[int, int, int]:
        """Generate a consistent color for a face ID using a hash function"""
        import hashlib
        # Use face_id to generate a consistent color
        hash_val = int(hashlib.md5(str(face_id).encode()).hexdigest()[:6], 16)
        r = (hash_val & 0xFF0000) >> 16
        g = (hash_val & 0x00FF00) >> 8
        b = hash_val & 0x0000FF
        return (b, g, r)  # OpenCV uses BGR

    def _find_prominent_face_in_segment(
        self,
        segment: Dict,
        face_positions: List[List[Dict]],
        frame_width: int
    ) -> Optional[Dict]:
        """Delegates to SpeechAndFaceAnalyzer's implementation"""
        return self.analyzer._find_prominent_face_in_segment(segment, face_positions, frame_width)
        
    def _validate_face_consistency(
        self,
        segment: Dict,
        face_positions: List[List[Dict]],
        face_pos: Dict,
        frame_width: int
    ) -> bool:
        """Delegates to SpeechAndFaceAnalyzer's implementation"""
        return self.analyzer._validate_face_consistency(segment, face_positions, face_pos, frame_width)
        
    def _find_alternate_face_position(
        self,
        current_pos: int,
        faces: List[Dict],
        frame_width: int
    ) -> Optional[Dict]:
        """Delegates to SpeechAndFaceAnalyzer's implementation"""
        return self.analyzer._find_alternate_face_position(current_pos, faces, frame_width)

# Create a singleton instance
video_processor = VideoProcessor()

def detect_face_and_crop(video_path: str, start: float, end: float) -> str:
    """Wrapper function for the video processor"""
    return video_processor.detect_face_and_crop(video_path, start, end)
