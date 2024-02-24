from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:3000/v1", api_key="not-needed")

completion = client.chat.completions.create(
    model="local-model", # this field is currently unused
    messages=[
        {
            "role": "user",
            "content": "Introduce yourself."
        }
    ]
)

print(completion.choices[0].message.content)