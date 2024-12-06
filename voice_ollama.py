import os
import pyaudio # Handles audio input/output.
from vosk import Model, KaldiRecognizer # speech recognition framework
import requests  # pip install requests
import pyttsx3  # Converts text to speech
import json  # Parses JSON data 

# Load Vosk model for offline speech recognition
def load_vosk_model():
    model_path = "models/vosk-model-small-en-us"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Vosk model not found! Download and place it in the 'models' directory.")
    return Model(model_path)

# Listen to microphone input and transcribe using Vosk
def listen_and_transcribe(model):
    rec = KaldiRecognizer(model, 16000)  # Vosk model and a sample rate of 16000 Hz
    audio = pyaudio.PyAudio() # Initializes the PyAudio instance for handling audio input/output.
    stream = audio.open(
        format=pyaudio.paInt16, # Opens an audio input stream
        channels=1, # Indicates mono audio (single channel).
        rate=16000, #The sample rate
        input=True, #Specifies that this stream is for audio input.
        frames_per_buffer=8192 # Sets the size of the buffer that holds audio data before being processed.
        )
    stream.start_stream()  # Begins the audio stream 

    print("Listening... Speak now.")
    while True:
        data = stream.read(4096) # Reads 4096 bytes of audio data (half of the buffer size)
        if rec.AcceptWaveform(data):   #Processes the audio chunk Returns True if the audio chunk is sufficient
            result = rec.Result()  #Retrieves the transcription result as a JSON string.
            text = eval(result).get('text', '')  # Converts the JSON string into a Python dictionary
            if text:
                return text

# Send the transcribed text to Ollama and get a response
def query_ollama(input_text):
    url = "http://localhost:11434/api/generate"  # 
    payload = {"model": "llama3.1:8b", "prompt": input_text}  # The input prompt provided by the user.
    try:
        response = requests.post(url, json=payload, stream=True)  # Stream the response, Enables streaming the response incrementally instead of loading it all at once.
        response.raise_for_status() # Raises an exception 

        full_response = ""    #Accumulates the complete response from the API.
        done = False           #

        # Iterate over the response in chunks
        for line in response.iter_lines(decode_unicode=True):            #It reads the response line by line
            if line:
                data = json.loads(line) #function takes a JSON-formatted string (the line) and converts it into a Python dictionary
                full_response += data.get("response", "")
                if data.get("done", False):   #Checks if the "done" field in the JSON is True, signaling the end of the response.
                    done = True
                    break

        if done:
            return full_response
        else:
            return "Ollama could not generate a valid response."

    except Exception as e:
        return f"Error: {e}"

# Convert Ollama's response to speech
def speak_text(text):
    engine = pyttsx3.init()  # This function initializes the text-to-speech engine
    engine.say(text) #The say() method takes the string passed as an argument (in this case, text) and adds it to the speech queue.
    engine.runAndWait() #tells the engine to start processing the queued text and actually speak it out loud.

# Main function to integrate all components
def main():
    try:
        # Load Vosk model
        model = load_vosk_model()

        while True:
            # Listen to user input
            user_input = listen_and_transcribe(model)
            print(f"You said: {user_input}")

            # Query Ollama API
            response = query_ollama(user_input)
            print(f"Ollama: {response}")

            # Speak Ollama's response
            speak_text(response)

    except KeyboardInterrupt:
        print("\nExiting. Goodbye!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
