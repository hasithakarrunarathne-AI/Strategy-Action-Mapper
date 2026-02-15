import os
from openai import OpenAI

key = "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"
print("OPENAI_API_KEY set?", bool(key))

client = OpenAI(api_key=key)

resp = client.chat.completions.create(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    messages=[{"role": "user", "content": "Reply with: OK"}],
    temperature=0,
)

print("API call success. Model replied:", resp.choices[0].message.content.strip())
