document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    const fileInput = form.querySelector('input[type="file"]');
    const successMessage = document.getElementById('success_message');
    const progressBar = document.querySelector('progress');
    const uploadedFilesHeader = document.getElementById('uploaded_files_header');
    const uploadedFilesTable = document.getElementById('uploaded_files_table');
    const uploadedFilesTableBody = uploadedFilesTable.getElementsByTagName('tbody')[0];
    const beginProcessingButton = document.getElementById('begin_processing_button');
    const finishedVideosButton = document.getElementById('view_finished_videos');
    const finishedCountSpan = document.getElementById('finished_count');

    // Only show video files in the file chooser
    fileInput.accept = '.mp4,.mkv,.avi,.mov';

    finishedVideosButton.addEventListener('click', function() {
        window.location.href = '/results';
    });

    // Function to process the next video in the queue
    async function processNextVideo() {
        try {
            const response = await fetch('/begin-processing', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                if (response.status === 404) {
                    // No more videos to process
                    beginProcessingButton.disabled = false;
                    beginProcessingButton.style.opacity = '1';
                    beginProcessingButton.style.cursor = 'pointer';
                    return false;
                } else if (response.status === 409) {
                    // A video is already being processed
                    alert('Please wait for the current video to finish processing.');
                    beginProcessingButton.disabled = false;
                    beginProcessingButton.style.opacity = '1';
                    beginProcessingButton.style.cursor = 'pointer';
                    return false;
                }
                throw new Error('Network response was not ok');
            }

            const result = await response.json();
            console.log('Processing started:', result);
            
            // Update the UI to show which file is being processed
            const rows = uploadedFilesTableBody.querySelectorAll('tr');
            for (const row of rows) {
                const fileNameCell = row.cells[0];
                const progressCell = row.cells[2];
                
                if (fileNameCell.textContent === result.filename) {
                    // Create and setup progress bar
                    const processingProgress = document.createElement('progress');
                    processingProgress.value = 0;
                    processingProgress.max = 100;
                    progressCell.innerHTML = '';
                    progressCell.appendChild(processingProgress);
                    
                    // Start polling for progress
                    startProgressPolling(result.filename, processingProgress, progressCell);
                    break;
                }
            }
            return true;
        } catch (error) {
            console.error('Error starting next video:', error);
            beginProcessingButton.disabled = false;
            beginProcessingButton.style.opacity = '1';
            beginProcessingButton.style.cursor = 'pointer';
            return false;
        }
    }

    // Separate function for progress polling
    function startProgressPolling(filename, progressBar, progressCell) {
        const pollProgress = setInterval(async () => {
            try {
                const progressResponse = await fetch(`/processing-progress/${filename}`);
                const progressData = await progressResponse.json();
                
                if (progressData.progress !== undefined) {
                    progressBar.value = progressData.progress;
                    
                    if (progressData.progress === 100) {
                        clearInterval(pollProgress);
                        progressCell.innerHTML = '<span class="completed">Completed</span>';
                        await updateFinishedCount();
                        // Refresh the file list since the completed file was removed
                        await loadUploadedFiles();
                        // Try to process next video
                        processNextVideo();
                    } else if (progressData.progress === 0 && progressData.error) {
                        clearInterval(pollProgress);
                        progressCell.innerHTML = '<span class="error">Failed</span>';
                        processNextVideo();
                    }
                }
            } catch (error) {
                console.error('Error checking progress:', error);
                clearInterval(pollProgress);
                progressCell.innerHTML = '<span class="error">Error</span>';
                processNextVideo();
            }
        }, 1000);
    }

    // Update the begin processing button click handler
    beginProcessingButton.addEventListener('click', function() {
        const hasFiles = uploadedFilesTableBody.querySelector('tr');
        if (!hasFiles) {
            console.error('No files to process');
            return;
        }
        beginProcessingButton.disabled = true;
        beginProcessingButton.style.opacity = '0.6';
        beginProcessingButton.style.cursor = 'not-allowed';
        processNextVideo();
    });

    form.addEventListener('submit', async function(event) {
        event.preventDefault();

        // Re-hide the success message if the user continues to upload more files
        successMessage.style.opacity = '0';
        successMessage.style.visibility = 'hidden';
        successMessage.style.display = 'none';
        successMessage.textContent = 'Files uploaded successfully!';  // Reset message text

        const formData = new FormData(form);

        // Verify file extensions
        const allowedExtensions = ['mp4', 'mkv', 'avi', 'mov'];
        for (const file of fileInput.files) {
            const fileExtension = file.name.split('.').pop().toLowerCase();
            if (!allowedExtensions.includes(fileExtension)) {
                alert(`File type not allowed: ${file.name}`);
                return;
            }
        }

        try {
            const xhr = new XMLHttpRequest();
            xhr.open('POST', '/upload', true);

            xhr.upload.onprogress = function(event) {
                if (event.lengthComputable) {
                    const percentComplete = (event.loaded / event.total) * 100;
                    progressBar.value = percentComplete;
                }
            };

            xhr.onload = function() {
                if (xhr.status === 200) {
                    // Display the success message with fade-in effect
                    successMessage.style.display = 'block';
                    setTimeout(() => {
                        successMessage.style.opacity = '1';
                        successMessage.style.visibility = 'visible';
                    }, 10);

                    // Complete the progress bar
                    progressBar.value = 100;

                    // Refresh the uploaded files table
                    loadUploadedFiles();
                } else {
                    const errorResponse = JSON.parse(xhr.responseText);
                    alert(`Error: ${errorResponse.error}`);
                    progressBar.value = 0;
                }
            };

            xhr.onerror = function() {
                console.error('Error:', xhr.statusText);
                progressBar.value = 0;
            };

            xhr.send(formData);
        } catch (error) {
            console.error('Error:', error);
            progressBar.value = 0;
        }
    });

    // Function to format file size
    function formatFileSize(bytes) {
        if (bytes >= 1073741824) { // 1GB
            return (bytes / 1073741824).toFixed(2) + ' GB';
        } else {
            return (bytes / 1048576).toFixed(2) + ' MB';  // Show in MB by default
        }
    }

    // Function to update finished videos count
    async function updateFinishedCount() {
        try {
            const finishedResponse = await fetch('/results', {
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            });
            const data = await finishedResponse.json();
            const videoCount = data.count;
            
            if (videoCount > 0) {
                finishedCountSpan.textContent = `${videoCount} finished video${videoCount > 1 ? 's' : ''}`;
                finishedCountSpan.classList.add('visible');
                finishedVideosButton.style.display = 'block';
            } else {
                finishedCountSpan.classList.remove('visible');
                finishedVideosButton.style.display = 'none';
            }
        } catch (error) {
            console.error('Error checking finished videos count:', error);
        }
    }

    // Function to load uploaded files and populate the table
    async function loadUploadedFiles() {
        try {
            const response = await fetch('/files');
            const files = await response.json();

            // Clear the table
            uploadedFilesTableBody.innerHTML = '';

            // Show/hide elements based on whether there are files
            const hasFiles = files.length > 0;
            uploadedFilesHeader.style.display = hasFiles ? 'block' : 'none';
            uploadedFilesTable.style.display = hasFiles ? 'table' : 'none';
            beginProcessingButton.style.display = hasFiles ? 'block' : 'none';
            beginProcessingButton.disabled = false;
            beginProcessingButton.style.opacity = '1';
            beginProcessingButton.style.cursor = 'pointer';

            // Show success message if there are no files and processing was happening
            if (!hasFiles && document.querySelector('progress[value="100"]')) {
                successMessage.textContent = 'All videos have been processed successfully!';
                successMessage.style.display = 'block';
                successMessage.style.opacity = '1';
                successMessage.style.visibility = 'visible';
            }

            if (hasFiles) {
                files.forEach(file => {
                    const row = uploadedFilesTableBody.insertRow();
                    const cellName = row.insertCell(0);
                    const cellSize = row.insertCell(1);
                    const cellProgress = row.insertCell(2);
                    
                    cellName.textContent = file.name;
                    cellSize.textContent = formatFileSize(file.size);
                    // Just show Ready status, don't check progress
                    cellProgress.innerHTML = '<span class="ready">Ready</span>';
                });
            }

            // Only update finished videos count
            await updateFinishedCount();
        } catch (error) {
            console.error('Error loading uploaded files:', error);
        }
    }

    // Initial load of uploaded files and finished count - but don't start processing
    loadUploadedFiles();
});