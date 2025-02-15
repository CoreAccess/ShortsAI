# Shorts AI (Short Clip Creator For YouTube & Tiktok)

This project is designed to take any YouTube video or video on your localhost and exact interesting 1 minute short clips from them, add subtitles to the clips and get them ready to upload to YouTube, all done programmatically with the help of AI

Additionally the goal is to have everything run locally (no need to pay for OpenAI). The goal is to avoid the costs of using APIs

## Features

-   **Video Download**: Using A Localhost Video or a YouTube Link (Coming Soon)
-   **Transcription**: Uses Whisper AI to transcribe the video.
-   **Highlight Extraction**: Uses Sentiment Analysis To Identify Interesting Parts of The Video
-   **Speaker Detection**: Detects speakers in the video (Coming Soon).
-   **Vertical Cropping**: Crops the highlighted sections vertically, making them perfect for shorts.

## Known Issues & Annoyances

-   **Coding Standards**: The code is a mess I'm focused on rapid prototyping, once I have the core functionality I will go back and use proper OOP standards and refactoring the code base
-   **Selecting Clips**: The method used to select intersting clips could be improved
-   **Face Tracking & Cropping**: Still needs work to track faces and crop correctly
-   **One Last Note**: Still in active development and each day changes a TON, keep this in mind please

## Installation

### Prerequisites

-   Python 3.12.9 (python --version)
-   FFmpeg
-   Nvidia GPU (Recommend 40 Series or Higher but 30 Series Should Also Work)
    -   Other GPUs not tested, but you might wind up using your CPU which is slow

### Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/CoreAccess/ShortsAI.git
    cd Shorts/AI
    ```

2. Create a virtual environment

```bash
python -m .venv .venv
```

3. Activate a virtual environment:

```bash
source venv/bin/activate # On Windows: .venv\Scripts\activate
```

4. Install the python dependencies:

```bash
pip install -r requirements.txt
```

5. Installing Torch:

This gets a bit tricky, Torch is required for the models to use your nVidia GPU but in order for it to work correctly you need to install the correct version of Torch for your GPU to do this run this in the terminal

```bash
nvidia-smi
```

After that, scroll up a bit and look for the CUDA version, in the table on the top right.

6. Finding The Right Version of Torch:

Visit the following link

```bash
https://pytorch.org/
```

Scroll down to the table with the orange squares and look for "Compute Platform" change this to your CUDA version from step 5. For instance if your CUDA version was 12.8, select CUDA 12.6 (closest one down from your version). Then copy the pip3 command displayed below that will look something like

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

When your running the program there should be an output or a log saying "Device set to use cuda", that's how you know if its working, if it says "Device set to use cpu" then unistall torch and try a different version

---

## Usage

1. Run the main script and add a local video (YouTube uploads coming soon):
    ```bash
    python app.py
    ```
