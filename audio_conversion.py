import azure.cognitiveservices.speech as speechsdk
from azure.ai.translation import TextTranslationClient
from azure.core.credentials import AzureKeyCredential

# Set up Azure Speech-to-Text API for Hebrew transcription
speech_config = speechsdk.SpeechConfig(subscription="YourAzureSubscriptionKey", region="YourRegion")
audio_config = speechsdk.audio.AudioConfig(filename="path_to_audio_file.wav")
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

# Recognize speech in Hebrew
result = speech_recognizer.recognize_once()
if result.reason == speechsdk.ResultReason.RecognizedSpeech:
    hebrew_text = result.text
    print(f"Recognized Hebrew Text: {hebrew_text}")

    # Use Azure Translator API to translate the Hebrew text into English
    translator_client = TextTranslationClient(endpoint="YourTranslatorAPIEndpoint", credential=AzureKeyCredential("YourTranslatorAPIKey"))
    translation = translator_client.translate(content=hebrew_text, to=["en"])
    english_text = translation[0].translations[0].text
    print(f"Translated English Text: {english_text}")

else:
    print("Speech recognition failed")