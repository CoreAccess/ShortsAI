from flask import Flask, jsonify, render_template, request, send_from_directory
from components.processing import process_video
import os
import threading
import signal
import sys
import torch
import logging
import traceback
from threading import Timer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['FINISHED_FOLDER'] = 'finished_videos/'
app.config['TEMP_FOLDER'] = 'temp_files/'
processing_progress = {}

class ProcessingWatchdog:
    def __init__(self, timeout, filename):
        self.timeout = timeout
        self.filename = filename
        self.timer = None 

    def start(self):
        self.timer = Timer(self.timeout, self.handle_timeout)
        self.timer.start()

    def stop(self):
        if self.timer:
            self.timer.cancel()

    def handle_timeout(self):
        logging.error(f"Processing timeout detected for {self.filename}")
        cleanup_resources()
        if self.filename in processing_progress:
            processing_progress[self.filename] = {"progress": 0, "error": True}

def cleanup_resources():
    """Clean up GPU and system resources"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logging.error(f"Error during resource cleanup: {e}")


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    logging.info("Shutting down server...")
    cleanup_resources()
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def cleanup_progress_dict():
    # Clean up completed or failed entries
    to_remove = []
    for filename, progress_data in processing_progress.items():
        if progress_data["progress"] == 100 or (progress_data["progress"] == 0 and progress_data["error"]):
            to_remove.append(filename)
    
    for filename in to_remove:
        del processing_progress[filename]


def init_app():
    # Reset all app state on startup
    global processing_progress
    processing_progress = {}

    # Create required directories if they don't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['FINISHED_FOLDER'], exist_ok=True)
    os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)
    
    logging.info("Application initialized - waiting for user to start processing")

# Initialize the application
init_app()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('file')

    # Allowed video extensions
    allowed_extensions = {'mp4', 'mkv', 'avi', 'mov'}

    for file in files:
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in allowed_extensions:
            return jsonify({'error': f'File type not allowed: {file.filename}'}), 400
        file_path = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        if os.path.exists(file_path):
            print(f'Skipping file {file.filename} as it already exists.')
            continue
        file.save(file_path)

    return jsonify({'message': 'Upload complete'})

@app.route('/begin-processing', methods=['POST'])
def begin_processing():
    try:
        # Don't start if there's already a video being processed
        if any(progress.get("progress", 0) > 0 and progress.get("progress", 0) < 100 
               for progress in processing_progress.values()):
            return jsonify({'error': 'A video is already being processed'}), 409
        
        # Clean up old entries
        cleanup_progress_dict()
        cleanup_resources()
        
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path) and filename.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
                if filename in processing_progress:
                    continue
                    
                filepath = os.path.abspath(file_path)
                processing_progress[filename] = {"progress": 0, "error": False}
                
                # Create watchdog with 30-minute timeout
                watchdog = ProcessingWatchdog(1800, filename)
                
                def process_video_with_monitoring(*args):
                    try:
                        watchdog.start()
                        process_video(*args)
                        watchdog.stop()
                    except Exception as e:
                        watchdog.stop()
                        logging.error(f"Unhandled exception in processing thread: {str(e)}")
                        logging.error(f"Stack trace: {traceback.format_exc()}")
                        cleanup_resources()
                
                thread = threading.Thread(target=process_video_with_monitoring, args=(
                    filepath, 
                    processing_progress,
                    app.config['TEMP_FOLDER'],
                    app.config['FINISHED_FOLDER']
                ))
                thread.daemon = True
                thread.start()
                
                logging.info(f"Started processing thread for {filename}")
                return jsonify({'status': 'Processing started', 'filename': filename})
        
        return jsonify({'error': 'No files available for processing'}), 404
    except Exception as e:
        logging.error(f"Error in begin_processing route: {str(e)}")
        logging.error(f"Stack trace: {traceback.format_exc()}")
        cleanup_resources()
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/processing-progress/<filename>')
def get_processing_progress(filename):
    # Only return progress for files that have explicitly started processing
    if filename not in processing_progress:
        return jsonify({"progress": 0, "error": False, "status": "ready"})
        
    # Check if file still exists in uploads folder
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        # If file doesn't exist but we have progress data showing it completed
        progress_data = processing_progress.get(filename, {})
        if progress_data.get('progress') == 100:
            cleanup_progress_dict()  # Clean up the progress dictionary
            return jsonify({"progress": 100, "error": False, "status": "completed"})
        # If file doesn't exist and wasn't completed, it may have failed
        cleanup_progress_dict()
        return jsonify({"progress": 0, "error": True, "status": "failed"})
    
    # Return normal progress if file still exists
    progress_data = processing_progress.get(filename, {"progress": 0, "error": False})
    return jsonify(progress_data)

@app.route('/view/<filename>')
def view_results(filename):
    # Get the results for the processed file
    filepath = os.path.abspath(os.path.join(os.getcwd(), app.config['UPLOAD_FOLDER'], filename))
    results_file = filepath + '.results.json'
    
    if not os.path.exists(results_file):
        return jsonify({'error': 'Results not found'}), 404
        
    with open(results_file, 'r') as f:
        results = f.read()
    
    return render_template('results.html', filename=filename, results=results)

@app.route('/files', methods=['GET'])
def list_files():
    # Don't trigger any processing, just list files
    files = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(file_path):
            # Include all files that aren't currently being processed
            if filename not in processing_progress:
                files.append({
                    'name': filename,
                    'size': os.path.getsize(file_path),
                    'status': 'ready'
                })
    return jsonify(files)

@app.route('/results')
def results():
    # Get list of finished videos
    finished_videos = []
    for filename in os.listdir(app.config['FINISHED_FOLDER']):
        if filename.endswith('.mp4'):  # Only include mp4 files
            file_path = os.path.join(app.config['FINISHED_FOLDER'], filename)
            file_size = os.path.getsize(file_path)
            finished_videos.append({
                'name': filename,
                'size': file_size,
                'path': f'/video/{filename}'
            })
    
    # If it's an AJAX request, return just the count
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'count': len(finished_videos)})
    return render_template('results.html', videos=finished_videos)

@app.route('/video/<filename>')
def serve_video(filename):
    return send_from_directory(app.config['FINISHED_FOLDER'], filename)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico')

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle any unhandled exception without crashing the server"""
    # Log the error and stack trace
    logging.error("Unhandled exception: %s", str(e))
    logging.error("Stack trace: %s", traceback.format_exc())
    
    # Clean up resources
    cleanup_resources()
    
    # If this is a background thread error, update progress
    thread_name = threading.current_thread().name
    if thread_name != "MainThread":
        for filename, progress in processing_progress.items():
            if progress.get("progress", 0) > 0 and progress.get("progress", 0) < 100:
                processing_progress[filename] = {"progress": 0, "error": True}
    
    # Return error response
    return jsonify({
        "error": "An internal server error occurred",
        "details": str(e) if app.debug else "Contact administrator for details"
    }), 500

if __name__ == '__main__': 
    app.run(debug=True)