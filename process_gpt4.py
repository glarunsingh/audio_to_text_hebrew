import os
from typing import List, Dict, Optional
import whisper
import openai
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pydub import AudioSegment

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define all fields in a single variable, including Phone Number
FIELDS: List[str] = [
    "Customer Name", "Address", "Phone Number", "Invoice Number", "Date",
    "VAT", "Delivery Cost", "Discount", "Total Amount",
    "Item Code", "Item Name / Description", "Quantity", "Price"
]
COLUMNS: List[str] = ["Audio File", "Transcribed Text", "English Transcription"] + FIELDS

# Load Whisper model
WHISPER_MODEL = whisper.load_model("large-v3-turbo")

def transcribe_audio(file_path: str) -> tuple[str, str]:
    """Transcribe and translate Hebrew audio file using Whisper, supporting .m4a with conversion."""
    # Check if file is .m4a and convert if needed
    if file_path.endswith(".m4a"):
        wav_path = file_path.replace(".m4a", ".wav")
        try:
            # Try direct transcription first
            hebrew_result = WHISPER_MODEL.transcribe(file_path, language="he", task="transcribe")
            english_result = WHISPER_MODEL.transcribe(file_path, language="he", task="translate")
        except Exception as e:
            print(f"Whisper failed on .m4a ({e}), converting to .wav...")
            audio = AudioSegment.from_file(file_path, format="m4a")
            audio.export(wav_path, format="wav")
            file_path = wav_path
            hebrew_result = WHISPER_MODEL.transcribe(file_path, language="he", task="transcribe")
            english_result = WHISPER_MODEL.transcribe(file_path, language="he", task="translate")
    else:
        hebrew_result = WHISPER_MODEL.transcribe(file_path, language="he", task="transcribe")
        english_result = WHISPER_MODEL.transcribe(file_path, language="he", task="translate")
    
    # Return Hebrew and English texts directly
    return hebrew_result["text"], english_result["text"]

def extract_entities_gpt4(text: str) -> Dict[str, Optional[str]]:
    """Extract entities using GPT-4 with a generic prompt."""
    prompt = f"""Analyze the following Hebrew text and extract relevant information into the specified fields: Customer Name, Address, Phone Number, Invoice Number, Date, VAT, Delivery Cost, Discount, Total Amount, Item Code, Item Name / Description, Quantity, Price. 

    - For Customer Name, identify the primary customer, distinguishing between roles like dealer or end customer if mentioned.
    - For Item Code and Item Name / Description, clearly separate distinct items if multiple are mentioned.
    - Include any additional details, such as installation notes, in the relevant fields (e.g., Item Name / Description).
    - Leave fields blank if the information is not found.
    - Return only the extracted information for these fields.

    Text: {text}"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    entities: Dict[str, Optional[str]] = {field: None for field in FIELDS}
    lines = response.choices[0].message.content.split("\n")
    for line in lines:
        for field in FIELDS:
            if field in line:
                value = line.split(field)[-1].strip(": ").strip()
                entities[field] = value if value else None
    return entities

def process_audio_folder_gpt4(folder_path: str, excel_file: str) -> pd.DataFrame:
    """Process audio files with GPT-4, include transcriptions, and save to Excel."""
    data: List[Dict[str, Optional[str]]] = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".mp3", ".wav", ".m4a")):
            file_path = os.path.join(folder_path, filename)
            print(f"GPT-4 Processing: {filename}")
            hebrew_text, english_text = transcribe_audio(file_path)
            entities = extract_entities_gpt4(hebrew_text)
            row = {
                "Audio File": filename,
                "Transcribed Text": hebrew_text,
                "English Transcription": english_text,
                **entities
            }
            data.append(row)
    df = pd.DataFrame(data, columns=COLUMNS)
    with pd.ExcelWriter(excel_file, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, sheet_name="GPT 4", index=False)
    return df

if __name__ == "__main__":
    folder_path: str = "/Users/ar20572127/Documents/GitHub/Audio_conversion/audio"  # Update this
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_file = f"entity_extraction_results_{timestamp}.xlsx"
    df_gpt4 = process_audio_folder_gpt4(folder_path, excel_file)
    print("GPT-4 DataFrame:\n", df_gpt4)