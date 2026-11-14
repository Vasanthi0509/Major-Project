import pyttsx3

class TextToSpeech:
    def __init__(self):
        self.engine = pyttsx3.init()
        # Set speed
        self.engine.setProperty('rate', 150)

    def say_objects(self, labels):
        if labels:
            text = f"I see {', '.join(labels)}"
            self.engine.say(text)
            self.engine.runAndWait()

# Usage: engine = TextToSpeech() -> engine.say_objects(['chair', 'door'])