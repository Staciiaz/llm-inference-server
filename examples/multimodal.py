import base64

from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:3000/v1", api_key="not-needed")

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "./SCR-20240224-ptwz.png"

# Getting the base64 string
base64_image = encode_image(image_path)

completion = client.chat.completions.create(
    model="local-model", # this field is currently unused
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What does this mean?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
)

print(completion.choices[0].message.content)