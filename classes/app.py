import os
from flask import jsonify, request
import threading
from components.processing import process_video

class AppClass:
    # Called when the class is first created
    def __init__(self, app):
        # Save the Flask app instance
        self.app = app

        # Used to store the progress of video processing
        self.processing_progress = {}

        # Create required directories if they don't exist
        os.makedirs(self.app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(self.app.config['FINISHED_FOLDER'], exist_ok=True)
        os.makedirs(self.app.config['TEMP_FOLDER'], exist_ok=True)

    # Called when the user uploads a file
    def upload(self):
        files = request.files.getlist('file')

        # Allowed video extensions
        allowed_extensions = {'mp4', 'mkv', 'avi', 'mov'}

        for file in files:
            file_extension = file.filename.split('.')[-1].lower()
            if file_extension not in allowed_extensions:
                return jsonify({'error': f'File type not allowed: {file.filename}'}), 400
            file_path = os.path.abspath(os.path.join(self.app.config['UPLOAD_FOLDER'], file.filename))
            if os.path.exists(file_path):
                print(f'Skipping file {file.filename} as it already exists.')
                continue
            file.save(file_path)

        return jsonify({'message': 'Upload complete'})
    
    # Called when the user requests to begin processing videos
    def begin_processing(self):
        try:
            # Don't start if there's already a video being processed
            if any(progress.get("progress", 0) > 0 and progress.get("progress", 0) < 100 
                for progress in self.processing_progress.values()):
                return jsonify({'error': 'A video is already being processed'}), 409
            
            # Clean up old entries
            self.cleanup_progress_dict()
            
            def process_next_video():
                with self.app.app_context():
                    for filename in os.listdir(self.app.config['UPLOAD_FOLDER']):
                        file_path = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
                        if os.path.isfile(file_path) and filename.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
                            if filename in self.processing_progress:
                                continue
                                
                            filepath = os.path.abspath(file_path)
                            self.processing_progress[filename] = {"progress": 0, "error": False}

                            def process_video_with_monitoring(*args):
                                try:
                                    process_video(*args) 
                                except Exception as e:
                                    print(f"Unhandled exception in processing thread: {str(e)}")
                                finally:
                                    process_next_video()  # Process the next video after the current one finishes
                            
                            thread = threading.Thread(target=process_video_with_monitoring, args=(
                                filepath, 
                                self.processing_progress,
                                self.app.config['TEMP_FOLDER'],
                                self.app.config['FINISHED_FOLDER']
                            ))
                            thread.daemon = True
                            thread.start()
                            
                            return jsonify({'status': 'Processing started', 'filename': filename})
                    return jsonify({'error': 'No files available for processing'}), 404
            
            return process_next_video()
        except Exception as e:
            return jsonify({'error': 'Internal server error'}), 500

    def cleanup_progress_dict(self):
        # Clean up completed or failed entries
        to_remove = []
        for filename, progress_data in self.processing_progress.items():
            if progress_data["progress"] == 100 or (progress_data["progress"] == 0 and progress_data["error"]):
                to_remove.append(filename)
        
        for filename in to_remove:
            del self.processing_progress[filename]

    def get_processing_progress(self, filename):
        # Only return progress for files that have explicitly started processing
        if filename not in self.processing_progress:
            return jsonify({"progress": 0, "error": False, "status": "ready"})
            
        # Check if file still exists in uploads folder
        file_path = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            # If file doesn't exist but we have progress data showing it completed
            progress_data = self.processing_progress.get(filename, {})
            if progress_data.get('progress') == 100:
                self.cleanup_progress_dict()  # Clean up the progress dictionary
                return jsonify({"progress": 100, "error": False, "status": "completed"})
            # If file doesn't exist and wasn't completed, it may have failed
            self.cleanup_progress_dict()
            return jsonify({"progress": 0, "error": True, "status": "failed"})
        
        # Return normal progress if file still exists
        progress_data = self.processing_progress.get(filename, {"progress": 0, "error": False})
        return jsonify(progress_data)
    
    def view_results(self, filename):
        # Get the results for the processed file
        filepath = os.path.abspath(os.path.join(os.getcwd(), self.app.config['UPLOAD_FOLDER'], filename))
        results_file = filepath + '.results.json'
        
        if not os.path.exists(results_file):
            return jsonify({'error': 'Results not found'}), 404
            
        with open(results_file, 'r') as f:
            results = f.read()
        
        return results
    
    def list_files(self):
        # Don't trigger any processing, just list files
        files = []
        for filename in os.listdir(self.app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                # Include all files that aren't currently being processed
                if filename not in self.processing_progress:
                    files.append({
                        'name': filename,
                        'size': os.path.getsize(file_path),
                        'status': 'ready'
                    })
        return jsonify(files)
    
    def results(self):
        # Get list of finished videos
        finished_videos = []
        for filename in os.listdir(self.app.config['FINISHED_FOLDER']):
            if filename.endswith('.mp4'):  # Only include mp4 files
                file_path = os.path.join(self.app.config['FINISHED_FOLDER'], filename)
                file_size = os.path.getsize(file_path)
                finished_videos.append({
                    'name': filename,
                    'size': file_size,
                    'path': f'/video/{filename}'
                })
        
        return finished_videos