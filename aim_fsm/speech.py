from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import webbrowser
import threading
import os
import logging

from .thesaurus import Thesaurus
from .evbase import Event
from .events import SpeechEvent


app = Flask('VEX AIM')
CORS(app)  # This will enable CORS for all routes

@app.route('/')
def serve_index():
    return send_from_directory('.', 'speech_listener.html')

@app.route('/api/speech-to-text', methods=['POST'])
def handle_speech_to_text():
    global speech_listener
    data = request.json
    speech_listener.handle_utterance(data['text'])
    response = {
        'message': 'Text received',
        'data': data
    }
    #print('Response received:', response)
    return jsonify(response)


class SpeechListener():
    def __init__(self, _robot, thesaurus=Thesaurus(), debug=False):
        global robot
        global speech_listener
        robot = _robot
        self.robot = robot
        self.thesaurus = thesaurus
        self.debug = debug
        self.flask_thread = None
        speech_listener = self

    def run_flask(self):
        # Suppress the default Flask logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        # Debug must be false to prevent duplicate tab:
        app.run(port=5000, debug=False)

    def load_listener_page(self):
        webbrowser.open_new_tab('http://127.0.0.1:5000/')
        print('Speech input is running in your browser.')

    def handle_utterance(self, utterance):
        print("Raw utterance: '%s'" % utterance)
        utterance = utterance.strip().lower()
        words = [self.thesaurus.lookup_word(w) for w in utterance.split(" ")]
        words = self.thesaurus.substitute_phrases(words)
        string = " ".join(words)
        print("Heard: '%s'" % string)
        evt = SpeechEvent(string,words)
        self.robot.erouter.post(evt)
        
    def start(self):
        if self.flask_thread:
            return
        self.flask_thread = threading.Thread(target=self.run_flask)
        self.flask_thread.start()
        self.robot.loop.call_later(1, self.load_listener_page)
