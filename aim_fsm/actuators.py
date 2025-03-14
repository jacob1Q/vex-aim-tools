import asyncio
import os

from gtts import gTTS
from google.cloud import texttospeech

from . import aim

class Actuator():
    class ActuatorLocked(Exception): pass
    class ActuatorNotHeld(Exception): pass

    def __init__(self, robot, name, stop_fn = lambda : None):
        self.robot = robot
        self.name = name
        self.holder = None
        self.started = False
        self.stop_fn = stop_fn

    def __repr__(self):
        return f"<Actuator {self.name}>"

    def lock(self, node):
        if self.holder is None:
            self.holder = node
            return True
        elif self.holder is node:
            return True
        else:
            raise self.ActuatorLocked(self)

    def unlock(self, node):
        if self.holder is node:
            self.holder = None
        else:
            raise self.ActuatorNotHeld()

    def unlock_if_held(self, node):
        "Needed if an external event shuts down a node that might have locked the actuator."
        if self.holder is node:
            self.holder = None

    def status_update(self): pass

    def complete(self):
        if self.holder:
            self.holder.complete(self)

class DriveActuator(Actuator):
    def __init__(self, robot):
        super().__init__(robot, 'drive')

    def stop(self):
        self.robot.robot0.stop_all_movement()

    def status_update(self):
        # Bad timing can cause a just-started node to complete prematurely;
        # must wait until robot is seen to be moving before considering
        # looking for a stopped-moving status.
        if self.robot.robot0.is_move_active() or self.robot.robot0.is_turn_active():
            self.started = True
        elif self.holder and self.started:
            self.holder.complete(self)
            self.holder = None
            self.started = False

    def turn(self, node, angle_rads, turn_speed=None):
        self.lock(node)
        self.started = False
        self.robot.turn(angle_rads, turn_speed=turn_speed)

    def forward(self, node, distance_mm, drive_speed=None):
        self.lock(node)
        self.started = False
        self.robot.forward(distance_mm, drive_speed=drive_speed)

    def sideways(self, node, distance_mm, drive_speed=None):
        self.lock(node)
        self.started = False
        self.robot.sideways(distance_mm, drive_speed=drive_speed)

    def move(self, node, distance_mm, angle_rads, drive_speed=None, turn_speed=None):
        self.lock(node)
        self.started = False
        self.robot.move(distance_mm, angle_rads, drive_speed=drive_speed, turn_speed=turn_speed)

class SoundActuator(Actuator):
    def __init__(self, robot):
        super().__init__(robot, 'sound')
        self.use_gcloud = True
        self.playing = False
        # Google text to speech setup:
        try:
            self.tts_client = texttospeech.TextToSpeechClient()
            self.tts_voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Journey-F",
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
            )
            self.tts_audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
        except:  # Cloud text-to-speech failed; use gTTs instead
            print('No Google Cloud credentials. Reverting to alternate speech synthesizer.')
            self.use_gcloud = False

    def status_update(self):
        if self.robot.robot0.is_sound_active():
            self.playing = True
        else:
            if self.playing is True:
                self.playing = False
                try:  # might fail if speech isn't up yet
                    self.robot.loop.call_later(1, self.robot.speech_listener.unpause)
                except:
                    pass
                self.complete()

    def say_text(self, node, text):
        self.lock(node)
        self.robot.loop.call_soon_threadsafe(self.launch_text_to_mp3, text)

    def launch_text_to_mp3(self, text):
        self.robot.loop.create_task(self.text_to_mp3(text))

    async def text_to_mp3(self, text):
        temp_dir = os.getenv('TEMP', '/tmp')
        speech_file_path = os.path.join(temp_dir, 'vex_speech.mp3')
        while True:
            if self.use_gcloud:
                synthesis_input = texttospeech.SynthesisInput(text=text)
                response = self.tts_client.synthesize_speech(
                    input = synthesis_input,
                    voice = self.tts_voice,
                    audio_config = self.tts_audio_config
                )
                with open(speech_file_path, 'wb') as out:
                    out.write(response.audio_content)
            else:
                tts = gTTS(text=text, lang='en')
                tts.save(speech_file_path)
            self.robot.speech_listener.pause()
            try:
                self.robot.robot0.play_sound_file(speech_file_path)
            except aim.invalid_sound_file_exception:   # file too long
                print("*** Speech too long. Truncating...")
                text = text[0:len(text)//2]
                continue
            return            

    def play_sound(self, node, sound, volume=100):
        self.lock(node)
        self.robot.robot0.play_sound(sound, volume)

    def play_sound_file(self, node, filepath):
        self.lock(node)
        self.robot.robot0.play_sound_file(filepath)


class KickActuator(Actuator):
    KICK_DURATION = 0.25 # seconds

    def __init__(self, robot):
        super().__init__(robot, 'kick')

    def kick(self, node, kicktype):
        self.lock(node)
        self.robot.robot0.kick(kicktype)
        self.robot.loop.call_soon_threadsafe(self.set_delayed_completion)

    def set_delayed_completion(self):
        self.robot.loop.create_task(self.delayed_completion())

    async def delayed_completion(self):
        await asyncio.sleep(self.KICK_DURATION)
        if self.holder:
            self.holder.complete(self)


class LEDsActuator(Actuator):
    def __init__(self, robot):
        super().__init__(robot, 'leds')
        self.NUM_LEDS = 6

    def stop(self):
        self.robot.robot0.clear_leds()

    def set_light_color(self, node, *args):
        self.lock(node)
        self.robot.robot0.set_light_color(*args)
