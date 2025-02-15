from flask import Flask, jsonify, render_template, request, send_from_directory
import logging
from classes.app import AppClass

app = Flask(__name__)

# Get the Werkzeug logger (which Flask uses)
log = logging.getLogger('werkzeug')

# Suppress access logs (INFO and below)
log.setLevel(logging.ERROR)  # Suppresses the HTTP request logs in the terminal

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['FINISHED_FOLDER'] = 'finished_videos/'
app.config['TEMP_FOLDER'] = 'temp_files/'

# Initialize the application
my_class = AppClass( app )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    return my_class.upload()

@app.route('/begin-processing', methods=['POST'])
def begin_processing():
    return my_class.begin_processing()

@app.route('/processing-progress/<filename>')
def get_processing_progress(filename):
    return my_class.get_processing_progress(filename)

@app.route('/view/<filename>')
def view_results(filename):
    results = my_class.view_results(filename)

    return render_template('results.html', filename=filename, results=results)

@app.route('/files', methods=['GET'])
def list_files():
    return my_class.list_files()

@app.route('/results')
def results():
    finished_videos = my_class.results()
    
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
    # Return error response
    return jsonify({
        "error": "An internal server error occurred",
        "details": str(e) if app.debug else "Contact administrator for details"
    }), 500

if __name__ == '__main__': 
    app.run(debug=True)