import whisper

# Load the model once (large-v3 recommended)
model = whisper.load_model("large-v3")

# Transcription to Hebrew
result_hebrew = model.transcribe("hebrew_audio1.mp3", language="he", task="transcribe")

# Translation to English
result_english = model.transcribe("hebrew_audio1.mp3", language="he", task="translate")

# Write both to one file
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("Hebrew Transcription:\n")
    f.write(result_hebrew["text"] + "\n\n")
    f.write("English Translation:\n")
    f.write(result_english["text"])