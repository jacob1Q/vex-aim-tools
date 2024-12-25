import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key:
    client = openai.OpenAI()
else:
    client = None

preamble = """
  You are an intelligent mobile robot.
  You answer questions as briefly as possible.
  To move forward by N millimeters, output the string "#forward N".
  To turn counter-clockwise by N degrees, output the string "#turn N", and use a negative value for clockwise turns.
"""

def gpt_query(query):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages = [
            {"role": "system", "content": preamble},
            {"role": "user", "content": query}
      ])
    return response.choices[0].message.content
