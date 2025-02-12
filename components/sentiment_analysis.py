from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from components.helpers import save_emotion_analysis
from datasets import Dataset
import torch

def analyze_emotions(transcription_segments, emotion_path):
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    emotion_pipeline = pipeline(
        "text-classification", 
        model=model, 
        tokenizer=tokenizer,
        device=device_str
    )

    texts = [segment["text"] for segment in transcription_segments]
    ds = Dataset.from_dict({"text": texts})

    results = emotion_pipeline(ds["text"])

    emotions = []
    for segment, result in zip(transcription_segments, results):
        label = result["label"].lower()
        score = result["score"]
        emotion_segment = {
            "start": segment["timestamp"][0],
            "end": segment["timestamp"][1],
            "label": label,
            "score": score,
            "text": segment["text"]
        }
        emotions.append(emotion_segment)

    save_emotion_analysis(emotions, emotion_path)
    
    return emotions
