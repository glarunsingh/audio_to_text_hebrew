import os
from typing import List, Dict, Optional
import whisper
from transformers import pipeline
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

# Define fields
ORDER_LEVEL_FIELDS: List[str] = [
    "Customer Name", "Address", "Invoice Number", "Date",
    "VAT", "Delivery Cost", "Discount", "Total Amount"
]
LINE_ITEM_FIELDS: List[str] = ["Item Code", "Item Name / Description", "Quantity", "Price"]
ALL_FIELDS: List[str] = ORDER_LEVEL_FIELDS + LINE_ITEM_FIELDS

# Load Whisper model
WHISPER_MODEL = whisper.load_model("large-v3")

def transcribe_audio(file_path: str) -> str:
    """Transcribe Hebrew audio file to text using Whisper."""
    transcription_file = file_path + ".txt"
    if os.path.exists(transcription_file):
        with open(transcription_file, "r", encoding="utf-8") as f:
            return f.read()
    result = WHISPER_MODEL.transcribe(file_path, language="he", task="transcribe")
    with open(transcription_file, "w", encoding="utf-8") as f:
        f.write(result["text"])
    return result["text"]

def extract_entities_dictalm(text: str) -> Dict[str, Optional[str]]:
    """Extract entities using DictaLM 2.0 with prompting, forcing CPU usage."""
    dictalm_model = pipeline(
        "text-generation",
        model="dicta-il/DictaLM2.0-instruct",
        trust_remote_code=True,
        device="cpu"  # Explicitly set to CPU
    )
    
    prompt = f"""בטקסט הבא בעברית, חלץ את הישויות הבאות: שם הלקוח, כתובת, מספר חשבונית, תאריך, מע"מ, עלות משלוח, הנחה, סכום כולל, קוד פריט, תיאור פריט, כמות, מחיר. השאר ריק אם אין מידע. טקסט: {text}"""
    
    with torch.no_grad():
        response = dictalm_model(
            prompt,
            max_length=200,
            num_return_sequences=1,
            truncation=True,  # Explicitly enable truncation
            pad_token_id=dictalm_model.tokenizer.eos_token_id  # Use EOS token for padding
        )[0]["generated_text"]

    entities: Dict[str, Optional[str]] = {field: None for field in ALL_FIELDS}
    lines = response.split("\n")
    for line in lines:
        for field in ALL_FIELDS:
            if field in line:
                value = line.split(field)[-1].strip(": ").strip()
                entities[field] = value if value and value != "ריק" else None
    
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return entities

def process_audio_folder_dictalm(folder_path: str, excel_file: str) -> pd.DataFrame:
    """Process audio files with DictaLM 2.0 and save to Excel."""
    data: List[Dict[str, Optional[str]]] = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".mp3", ".wav")):
            file_path = os.path.join(folder_path, filename)
            print(f"DictaLM 2.0 Processing: {filename}")
            hebrew_text = transcribe_audio(file_path)
            entities = extract_entities_dictalm(hebrew_text)
            row = {"Audio File": filename, **entities}
            data.append(row)
    df = pd.DataFrame(data, columns=["Audio File"] + ALL_FIELDS)
    with pd.ExcelWriter(excel_file, engine="openpyxl", mode="a" if os.path.exists(excel_file) else "w") as writer:
        df.to_excel(writer, sheet_name="DictaLM 2.0", index=False)
    return df

if __name__ == "__main__":
    folder_path: str = "/Users/ar20572127/Documents/audio_files"  # Update this
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_file = f"entity_extraction_results_{timestamp}.xlsx"
    df_dictalm = process_audio_folder_dictalm(folder_path, excel_file)
    print("DictaLM 2.0 DataFrame:\n", df_dictalm)