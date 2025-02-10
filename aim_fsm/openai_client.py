import os
import re
import cv2
import base64
import openai

from .events import OpenAIEvent

default_preamble = """
  You are an intelligent mobile robot named Celeste.
  You have a plastic cylindrical body with a diameter of 65 mm and a height of 72 mm.
  You have three omnidirectional wheels and a forward-facing camera.
  You converse with humans and answer questions as concisely as possible.
  Pronounce "AprilTag-1.a" as "April Tag 1-A", and similarly for any word of form "AprilTag-N.x".
  Pronounce "OrangeBarrel.a" as "Orange Barrel A", pronounce "BlueBarrel.b" as "Blue Barrel B", and similarly for other barrel designators.
  Remember to be concise in your answers.
"""

class OpenAIClient():
    def __init__(self, robot, model='gpt-4o'):
        self.robot = robot
        self.model = model
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if openai.api_key:
            self.client = openai.OpenAI()
        else:
            print("*** No OPENAI_API_KEY provided.  GPT will not be available.")
            self.client = None
        self.set_preamble(default_preamble)

    def set_preamble(self, preamble):
        self.messages = [
            {'role': 'system', 'content': preamble}
        ]

    def query(self, query_text):
        self.messages.append({'role': 'system', 'content': self.robot.world_map.get_prompt()})
        self.messages.append({'role': 'user', 'content': query_text})
        self.robot.loop.call_soon_threadsafe(self.launch_openai_query)

    def camera_query(self, query_text):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        result, encimg = cv2.imencode('.jpg', self.robot.camera_image, encode_param)
        base64_image = base64.b64encode(encimg).decode('utf-8')
        self.messages.append(
            {'role' : 'user',
             'content' : [
                 {'type': 'text', 'text': query_text },
                 {'type': 'image_url',
                  'image_url': {'url': f'data:image/jpeg;base64,{base64_image}'}}
             ]})
        self.robot.loop.call_soon_threadsafe(self.launch_openai_query)

    def send_camera_image(self, instruction=None):
        default_instruction = 'Here is the current camera image. Please go ahead and reply to the last request.' 
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        result, encimg = cv2.imencode('.jpg', self.robot.camera_image, encode_param)
        base64_image = base64.b64encode(encimg).decode('utf-8')
        self.messages.append(
            {'role' : 'user',
             'content' : [
                 {'type': 'text', 'text': instruction or default_instruction},
                 {'type': 'image_url',
                  'image_url': {'url': f'data:image/jpeg;base64,{base64_image}'}}
             ]})
        self.robot.loop.call_soon_threadsafe(self.launch_openai_query)

    def launch_openai_query(self):
        self.robot.loop.create_task(self.openai_query())

    async def openai_query(self):
        if self.client is None:
            return
        response = self.client.chat.completions.create(
            model = self.model,
            messages = self.messages
        )
        answer = response.choices[0].message.content
        self.messages.append({'role': 'assistant', 'content': answer})
        # remove LaTeX brackets from response
        cleaned_answer = re.sub(r'\\[\[\]\(\)]', '', answer)
        event = OpenAIEvent(cleaned_answer)
        self.robot.erouter.post(event)
