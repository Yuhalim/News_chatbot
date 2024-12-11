import time
import requests
from transformers import pipeline
from apscheduler.schedulers.background import BackgroundScheduler
from gtts import gTTS
import pygame
import tempfile
import cv2  # Import OpenCV for face detection
import speech_recognition as sr  # Import the speech_recognition library

# Initialize pygame mixer
pygame.mixer.init()

# News API configuration
NEWS_API_KEY = '9bee8e3e891b4821a5e3e5236aea7efa'
NEWS_BASE_URL = "https://newsapi.org/v2/top-headlines"
DEFAULT_COUNTRY = "us"

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Initialize recognizer
recognizer = sr.Recognizer()

# Function to fetch news articles
def fetch_news(country=DEFAULT_COUNTRY, limit=5):
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

# Function to summarize a news article
def summarize_news(article):
    try:
        if len(article) < 50:  # Minimum length check
            return article  # If the content is too short, return as is.

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

# Function to handle voice input using Google Speech Recognition
def recognize_speech(timeout=10):
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        try:
            audio = recognizer.listen(source, timeout=timeout)  # Listen to the microphone
            print("Recognizing...")
            command = recognizer.recognize_google(audio)  # Use Google Speech Recognition
            return command.lower()
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            return "unrecognized"
        except sr.RequestError:
            print("Error with the speech recognition service.")
            return "error"
        except Exception as e:
            return "error"

# Function to generate and play speech using gTTS
def speak_text(text):
    try:
        # Generate speech using gTTS
        tts = gTTS(text=text, lang='en')

        # Create a temporary file to save the speech audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            audio_path = temp_file.name
            tts.save(audio_path)

        # Use pygame to play the audio from the temporary file
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()

        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)    
    except Exception as e:
        print(f"Error with speech: {e}")

# Function to process news articles
def process_news():
    speak_text("Fetching latest headlines... please wait")
    articles = fetch_news()
    if not articles:
        print("No news available.")
        speak_text("Sorry, no news is available right now.")
        return

    for article in articles:
        # Check for interruption
        if pygame.mixer.music.get_busy():
            command = recognize_speech().lower()
            if "stop" in command:
                print("User requested to stop. Returning to detection mode.")
                speak_text("Okay, stopping the news update.")
                return

        title = article.get("title", "No Title Available")
        description = article.get("description") or article.get("content") or "No additional content available."
        
        # Speak the headline
        print(f"Headline: {title}")
        speak_text(f"Headline: {title}")
        
        # Check for interruption
        if pygame.mixer.music.get_busy():
            command = recognize_speech().lower()
            if "stop" in command:
                print("User requested to stop. Returning to detection mode.")
                speak_text("Okay, stopping the news update.")
                return
        
        # Speak the summary
        summary = summarize_news(description)
        print(f"Summary: {summary}")
        speak_text(f" {summary}")
        print("-" * 50)
        time.sleep(2)  # Pause between articles

        # Check for interruption
        if pygame.mixer.music.get_busy():
            command = recognize_speech().lower()
            if "stop" in command:
                print("User requested to stop. Returning to detection mode.")
                speak_text("Okay, stopping the news update.")
                return

# Scheduler to update news periodically
scheduler = BackgroundScheduler()

def fetch_periodic_news():
    print("\nPeriodic Update: Latest News Headlines")
    process_news()

scheduler.add_job(fetch_periodic_news, "interval", minutes=10)
scheduler.start()

# Function to detect motion or face using the camera
def camera_detection():
    cap = cv2.VideoCapture(0)
    first_frame = None

    print("System initialized. Observing for user presence...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if first_frame is None:
            first_frame = gray
            continue

        # Detect motion
        delta_frame = cv2.absdiff(first_frame, gray)
        thresh_frame = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        person_detected = False

        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                continue

            person_detected = True
            # Highlight the detected area
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Camera Feed", frame)

        if person_detected:
            print("A person has been detected! Starting the system...")
            speak_text("Hello!")
            time.sleep(1)
            speak_text("Would you like to hear the latest news?")
            while True:
                command = recognize_speech().lower()
                if "yes" in command:
                    process_news()
                    break
                elif "no" in command or "stop" in command:
                    print("Okay, stopping the news updates. Have a great day!")
                    speak_text("Okay, stopping the news updates. Have a great day!")
                    break
                elif command in ["unrecognized", "error"]:
                    print("I didn't catch that. Please say 'yes' or 'no'.")
                    speak_text("I didn't catch that. Please say 'yes' or 'no'.")
                else:
                    print("Unrecognized response. Please try again.")
                    speak_text("Unrecognized response. Please try again.")

            print("Returning to detection mode...")
            speak_text("Returning to detection mode.")
            # Reset first frame to continue monitoring
            first_frame = gray
        else:
            print("No person detected. Observing...")

        # Exit loop if 'q' is pressed
        key = cv2.waitKey(1)
        if key == ord("q"):
            print("Exiting the system.")
            speak_text("Goodbye! Have a great day.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    speak_text("Starting the AI news assistant.")
    camera_detection()
