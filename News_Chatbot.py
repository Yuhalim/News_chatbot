import time
import requests
from transformers import pipeline
from apscheduler.schedulers.background import BackgroundScheduler
from gtts import gTTS
import pygame
import tempfile
import cv2
import speech_recognition as sr
import threading 
import keyboard


NEWS_API_KEY = '9bee8e3e891b4821a5e3e5236aea7efa'
NEWS_BASE_URL = "https://newsapi.org/v2/top-headlines"
DEFAULT_COUNTRY = "us"


pygame.mixer.init()
stop_flag = threading.Event()
recognizer = sr.Recognizer()
# Initialize summarizer (adjust for GPU or CPU)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)


def fetch_news(country=DEFAULT_COUNTRY, limit=5):
    """Fetch news articles from the News API."""
    articles = []
    page = 1
    while len(articles) < limit:

        params = {"country": country, "apiKey": NEWS_API_KEY, "page": page, "pageSize": 100}
        response = requests.get(NEWS_BASE_URL, params=params)
        if response.status_code == 200:
            new_articles = response.json().get("articles", [])
            if not new_articles:
                break
            articles.extend(new_articles)
            page += 1
        else:
            print(f"Error fetching news: {response.status_code}")
            break
    return articles[:limit]

def summarize_news(article):
    """Summarize a news article."""
    try:
        if len(article) < 50:
            return article
        max_input_length = 1024
        article_chunks = [article[i:i + max_input_length] for i in range(0, len(article), max_input_length)]
        
        summaries = []
        for chunk in article_chunks:
            max_length = min(len(chunk.split()), 100)
            summary = summarizer(chunk, max_length=max_length, min_length=10, do_sample=False)
            summaries.append(summary[0]["summary_text"])
        return " ".join(summaries)
    except Exception as e:
        print(f"Error summarizing article: {e}")
        return "Could not summarize the article."

def recognize_speech(timeout=15):
    """Recognize speech input with improved handling."""
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
            # print("Recognizing...")
            result = recognizer.recognize_google(audio, show_all=True)
            if result:
                best_match = result["alternative"][0]["transcript"].lower()
                confidence = result["alternative"][0]["confidence"]
                print(f"Recognized: {best_match} (Confidence: {confidence})")
                if confidence > 0.8:
                    return best_match
            return "unrecognized"
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            return "unrecognized"
        except sr.RequestError as e:
            print(f"Error with the speech recognition service: {e}")
            return "error"
        except Exception as e:
            print(f"Unexpected error: {e}")
            return "error"


def speak_text(text):
    """Convert text to speech and play it with interruption support."""
    global stop_flag
    stop_flag.clear()  # Reset the stop flag

    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            audio_path = temp_file.name
            tts.save(audio_path)

        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()

        # Monitor for the stop flag during playback
        while pygame.mixer.music.get_busy():
            if stop_flag.is_set():  # If stop command detected
                pygame.mixer.music.stop()
                print("Playback interrupted by user.")
                return
            pygame.time.Clock().tick(10)

    except Exception as e:
        print(f"Error with speech: {e}")


def recognize_stop_command():
    """Continuously listen for a stop command."""
    global stop_flag
    while not stop_flag.is_set():
        command = recognize_speech(timeout=5).lower()
        if "stop" in command:
            print("Stop command detected.")
            stop_flag.set()  # Signal to stop the news updates
            return

def process_news():
    """Fetch, summarize, and present news to the user with interruption support."""
    speak_text("Fetching latest headlines... please wait")
    articles = fetch_news()
    if not articles:
        print("No news available.")
        speak_text("Sorry, no news is available right now.")
        return

    for article in articles:
        if stop_flag.is_set():
            print("User requested to stop. Returning to detection mode.")
            speak_text("Okay, stopping the news update.")
            time.sleep(5)  # Wait before returning to detection
            return

        title = article.get("title", "No Title Available")
        description = article.get("description") or article.get("content") or "No additional content available."

        # Start a thread to listen for stop command
        thread = threading.Thread(target=recognize_stop_command, daemon=True)
        thread.start()

        # Speak the headline
        print(f"Headline: {title}")
        speak_text(f"Headline: {title}")

        if stop_flag.is_set():
            return  # Stop if the flag is set

        # Speak the summary
        summary = summarize_news(description)
        print(f"Summary: {summary}")
        speak_text(summary)

        print("-" * 50)
        time.sleep(2)

scheduler = BackgroundScheduler()


def fetch_periodic_news():
    print("\nPeriodic Update: Latest News Headlines")
    process_news()

scheduler.add_job(fetch_periodic_news, "interval", minutes=10)
scheduler.start()
# 
# 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def camera_detection():
    """Continuously detect human faces and interact when a face is present."""
    cap = cv2.VideoCapture(0)
    print("System initialized. Observing for user presence...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Detect faces using Haar Cascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:  # If a face is detected
            print("Human face detected! Starting interaction...")
            speak_text("Hello! Would you like to hear the latest news?")
            
            while True:  # Interaction loop
                command = recognize_speech().lower()
                
                if "yes" in command:
                    process_news()
                elif "no" in command or "stop" in command:
                    print("Stopping news updates and returning to detection mode.")
                    speak_text("Okay, stopping the news updates.")
                    break
                elif command in ["unrecognized", "error"]:
                    print("I didn't catch that. Please say 'yes' or 'no'.")
                    speak_text("I didn't catch that. Please say 'yes' or 'no'.")
                else:
                    print("Unrecognized response. Please try again.")
                    speak_text("Unrecognized response. Please try again.")

                # Check if the face is still present during interaction
                ret, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(faces) == 0:  # No face detected
                    print("Face no longer detected.")
                    speak_text("Returning to monitoring mode.")
                    break

        else:
            print("No face detected. Monitoring...")
            # speak_text("No face detected. Waiting for a user.")
            time.sleep(1)  # Add a delay to reduce CPU usage

        # Show the frame with bounding boxes around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Camera Feed", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):  # Exit the program when 'q' is pressed
            print("Exiting the system.")
            speak_text("Goodbye! Have a great day.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    speak_text("Starting the AI news assistant.")
    camera_detection()


