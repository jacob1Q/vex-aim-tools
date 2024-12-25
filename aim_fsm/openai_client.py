import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key:
    client = openai.OpenAI()
else:
    client = None

preamble = "You are an intelligent mobile robot. You answer questions as briefly as possible."

def gpt_query(query):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages = [
            {"role": "system", "content": preamble},
            {"role": "user", "content": query}
      ])
    print(response)
    return response.choices[0].message.content
