import os
from typing import List, Dict, Optional
import whisper
from transformers import pipeline
import openai
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define fields for DataFrame structure
ORDER_LEVEL_FIELDS: List[str] = [
    "Customer Name", "Address", "Invoice Number", "Date",
    "VAT", "Delivery Cost", "Discount", "Total Amount"
]
LINE_ITEM_FIELDS: List[str] = ["Item Code", "Item Name / Description", "Quantity", "Price"]
ALL_FIELDS: List[str] = ORDER_LEVEL_FIELDS + LINE_ITEM_FIELDS

# Load Whisper model once for all functions
WHISPER_MODEL = whisper.load_model("large-v3")

# --- Common Utility Function ---
def transcribe_audio(file_path: str) -> str:
    """Transcribe Hebrew audio file to text using Whisper model."""
    result = WHISPER_MODEL.transcribe(file_path, language="he", task="transcribe")
    return result["text"]

# --- HeRo Block ---
def extract_entities_hero(text: str) -> Dict[str, Optional[str]]:
    """Extract entities from Hebrew text using HeRo NER model with heuristic rules."""
    ner_model = pipeline("ner", model="HeNLP/HeRo", tokenizer="HeNLP/HeRo", aggregation_strategy="simple")
    ner_results = ner_model(text)
    entities: Dict[str, Optional[str]] = {field: None for field in ALL_FIELDS}
    address_parts: List[str] = []
    item_parts: List[str] = []

    # Process NER results
    for entity in ner_results:
        word = entity["word"]
        entity_type = entity["entity_group"]
        if entity_type == "PER":
            entities["Customer Name"] = word
        elif entity_type == "LOC":
            address_parts.append(word)

    # Apply heuristic rules for non-NER entities
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

def process_audio_folder_hero(folder_path: str) -> pd.DataFrame:
    """Process audio files in folder using HeRo and return DataFrame."""
    data: List[Dict[str, Optional[str]]] = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".mp3", ".wav")):
            file_path = os.path.join(folder_path, filename)
            print(f"HeRo Processing: {filename}")
            hebrew_text = transcribe_audio(file_path)
            entities = extract_entities_hero(hebrew_text)
            row = {"Audio File": filename, **entities}
            data.append(row)
    return pd.DataFrame(data, columns=["Audio File"] + ALL_FIELDS)

# --- DictaLM 2.0 Block ---
def extract_entities_dictalm(text: str) -> Dict[str, Optional[str]]:
    """Extract entities from Hebrew text using DictaLM 2.0 with prompting."""
    dictalm_model = pipeline("text-generation", model="dicta-il/DictaLM2.0-instruct", trust_remote_code=True)
    prompt = f"""בטקסט הבא בעברית, חלץ את הישויות הבאות: שם הלקוח, כתובת, מספר חשבונית, תאריך, מע"מ, עלות משלוח, הנחה, סכום כולל, קוד פריט, תיאור פריט, כמות, מחיר. השאר ריק אם אין מידע. טקסט: {text}"""
    response = dictalm_model(prompt, max_length=500, num_return_sequences=1)[0]["generated_text"]

    entities: Dict[str, Optional[str]] = {field: None for field in ALL_FIELDS}
    lines = response.split("\n")
    for line in lines:
        for field in ALL_FIELDS:
            if field in line:
                value = line.split(field)[-1].strip(": ").strip()
                entities[field] = value if value and value != "ריק" else None
    return entities

def process_audio_folder_dictalm(folder_path: str) -> pd.DataFrame:
    """Process audio files in folder using DictaLM 2.0 and return DataFrame."""
    data: List[Dict[str, Optional[str]]] = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".mp3", ".wav")):
            file_path = os.path.join(folder_path, filename)
            print(f"DictaLM 2.0 Processing: {filename}")
            hebrew_text = transcribe_audio(file_path)
            entities = extract_entities_dictalm(hebrew_text)
            row = {"Audio File": filename, **entities}
            data.append(row)
    return pd.DataFrame(data, columns=["Audio File"] + ALL_FIELDS)

# --- GPT-4 Block ---
def extract_entities_gpt4(text: str) -> Dict[str, Optional[str]]:
    """Extract entities from Hebrew text using GPT-4 via API with prompting."""
    prompt = f"""From the following Hebrew text, extract the following entities: Customer Name, Address, Invoice Number, Date, VAT, Delivery Cost, Discount, Total Amount, Item Code, Item Name / Description, Quantity, Price. Leave blank if not found. Text: {text}"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    entities: Dict[str, Optional[str]] = {field: None for field in ALL_FIELDS}
    lines = response.choices[0].message["content"].split("\n")
    for line in lines:
        for field in ALL_FIELDS:
            if field in line:
                value = line.split(field)[-1].strip(": ").strip()
                entities[field] = value if value else None
    return entities

def process_audio_folder_gpt4(folder_path: str) -> pd.DataFrame:
    """Process audio files in folder using GPT-4 and return DataFrame."""
    data: List[Dict[str, Optional[str]]] = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".mp3", ".wav")):
            file_path = os.path.join(folder_path, filename)
            print(f"GPT-4 Processing: {filename}")
            hebrew_text = transcribe_audio(file_path)
            entities = extract_entities_gpt4(hebrew_text)
            row = {"Audio File": filename, **entities}
            data.append(row)
    return pd.DataFrame(data, columns=["Audio File"] + ALL_FIELDS)

# --- Save to Excel ---
def save_to_excel(df_hero: pd.DataFrame, df_dictalm: pd.DataFrame, df_gpt4: pd.DataFrame) -> None:
    """Save DataFrames to separate sheets in a single Excel file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"entity_extraction_results_{timestamp}.xlsx"
    
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_hero.to_excel(writer, sheet_name="HeRo", index=False)
        df_dictalm.to_excel(writer, sheet_name="DictaLM 2.0", index=False)
        df_gpt4.to_excel(writer, sheet_name="GPT 4", index=False)
    
    print(f"Results saved to {output_file}")

# --- Main Execution ---
if __name__ == "__main__":
    folder_path: str = "/Users/ar20572127/Documents/GitHub/Audio_conversion/audio"  # Update this with your actual path
    
    # Process each model independently
    df_hero = process_audio_folder_hero(folder_path)
    df_dictalm = process_audio_folder_dictalm(folder_path)
    df_gpt4 = process_audio_folder_gpt4(folder_path)
    
    # Save results
    save_to_excel(df_hero, df_dictalm, df_gpt4)