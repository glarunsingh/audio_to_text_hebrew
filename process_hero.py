import os
from typing import List, Dict, Optional
import whisper
from transformers import pipeline
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

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

def extract_entities_hero(text: str) -> Dict[str, Optional[str]]:
    """Extract entities using HeRo NER with heuristic rules."""
    ner_model = pipeline("ner", model="HeNLP/HeRo", tokenizer="HeNLP/HeRo", aggregation_strategy="simple")
    ner_results = ner_model(text)
    entities: Dict[str, Optional[str]] = {field: None for field in ALL_FIELDS}
    address_parts: List[str] = []
    item_parts: List[str] = []

    for entity in ner_results:
        word = entity["word"]
        entity_type = entity["entity_group"]
        if entity_type == "PER":
            entities["Customer Name"] = word
        elif entity_type == "LOC":
            address_parts.append(word)

    sentences = text.split(",")
    for sentence in sentences:
        sentence = sentence.strip()
        if any(kw in sentence for kw in ["רחוב", "כתובת", "סניף"]):
            address_parts.append(sentence)
        elif any(kw in sentence for kw in ["יחידות", "דגם", "מוצר"]):
            item_parts.append(sentence)
        elif "₪" in sentence:
            entities["Price"] = sentence if not entities["Price"] else entities["Price"]
        elif any(word.isdigit() and int(word) <= 10 for word in sentence.split()):
            entities["Quantity"] = next((word for word in sentence.split() if word.isdigit() and int(word) <= 10), None)

    entities["Address"] = " ".join(set(address_parts)) if address_parts else None
    entities["Item Name / Description"] = " ".join(set(item_parts)) if item_parts else None
    return entities

def process_audio_folder_hero(folder_path: str, excel_file: str) -> pd.DataFrame:
    """Process audio files with HeRo and save to Excel."""
    data: List[Dict[str, Optional[str]]] = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".mp3", ".wav")):
            file_path = os.path.join(folder_path, filename)
            print(f"HeRo Processing: {filename}")
            hebrew_text = transcribe_audio(file_path)
            entities = extract_entities_hero(hebrew_text)
            row = {"Audio File": filename, **entities}
            data.append(row)
    df = pd.DataFrame(data, columns=["Audio File"] + ALL_FIELDS)
    with pd.ExcelWriter(excel_file, engine="openpyxl", mode="a" if os.path.exists(excel_file) else "w") as writer:
        df.to_excel(writer, sheet_name="HeRo", index=False)
    return df

if __name__ == "__main__":
    folder_path: str = "/Users/ar20572127/Documents/GitHub/Audio_conversion/audio"  # Update this
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_file = f"entity_extraction_results_{timestamp}.xlsx"
    df_hero = process_audio_folder_hero(folder_path, excel_file)
    print("HeRo DataFrame:\n", df_hero)