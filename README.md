# Shorts AI (1 Minute Short Clip Creator For YouTube & Tiktok)

This project is designed to take any YouTube video or video on your localhost and exact interesting 1 minute short clips from them, add subtitles to the clips and get them ready to upload to YouTube, all done programmatically with the help of AI

Additionally the goal is to have everything run locally (no need to pay for OpenAI). The goal is to avoid the costs of using APIs

## Features

-   **Video Download**: Using A Localhost Video or a YouTube Link (Coming Soon)
-   **Transcription**: Uses Whisper AI to transcribe the video.
-   **Highlight Extraction**: Uses Sentiment Analysis To Identify Interesting Parts of The Video
-   **Speaker Detection**: Detects speakers in the video (Coming Soon).
-   **Vertical Cropping**: Crops the highlighted sections vertically, making them perfect for shorts.

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

---

## Usage

1. Run the main script and add a local video (YouTube uploads coming soon):
    ```bash
    streamlit run main.py
    ```
