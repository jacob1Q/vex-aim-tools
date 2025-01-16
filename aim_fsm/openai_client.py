import os
import re
import openai

from .events import OpenAIEvent

preamble = """
  You are an intelligent mobile robot named Celeste.
  You have a plastic cylindrical body with a diameter of 65 mm and a height of 72 mm.
  You have three omnidirectional wheels and a forward-facing camera.
  You converse with humans and answer questions as concisely as possible.
  Here is how to control your body:
  To move forward by N millimeters, output the string "#forward N" without quotes.
  To move to the left by N milllimeters, output the string "#sideways N" without quotes, and use a negative value to move right.
  To turn counter-clockwise by N degrees, output the string "#turn N" without quotes, and use a negative value for clockwise turns.
  To turn toward object X, output the string "#turntoward X" without quotes.
  To pick up object X, output the string "#pickup X" without quotes.
  To drop an object, output the string "#drop" without quotes.
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
        self.messages = [
            {'role': 'system', 'content': preamble}
        ]

    def query(self, query_text):
        self.messages.append({'role': 'system', 'content': self.robot.world_map.get_prompt()})
        self.messages.append({'role': 'user', 'content': query_text})
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
