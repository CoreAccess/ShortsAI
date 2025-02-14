import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import torch

def analyze_emotions(conversations):
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
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

    # Create dataset from conversation texts
    text_dataset = [{"text": segment[2]} for segment in conversations]
    ds = Dataset.from_list(text_dataset)

    # Get emotion results
    results = emotion_pipeline(ds["text"])

    # Combine original segments with emotion analysis
    analyzed_segments = []
    for (start_time, end_time, text), result in zip(conversations, results):
        analyzed_segments.append({
            "start": start_time,
            "end": end_time,
            "text": text,
            "emotion": result["label"].lower(),
            "emotion_score": result["score"]
        })

    return analyzed_segments
