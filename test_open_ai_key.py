import os
from openai import OpenAI

key = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("OPENAI_API_KEY set?", bool(key))

client = OpenAI(api_key=key)

resp = client.chat.completions.create(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    messages=[{"role": "user", "content": "Reply with: OK"}],
    temperature=0,
)

print("API call success. Model replied:", resp.choices[0].message.content.strip())
