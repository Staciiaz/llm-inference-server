from dotenv import load_dotenv

from src import models

if __name__ == '__main__':
    load_dotenv()

    messages = [
        {"role": "user", "content": "Give me one concise sentence on what is python programming."},
    ]

    # model = models.MockChatModel()
    # model = models.Llama("meta-llama/Llama-2-7b-chat-hf")
    model = models.Gemma("google/gemma-2b-it")
    # model = models.Mistral("mistralai/Mistral-7B-Instruct-v0.2")
    # prompt = messages[-1]["content"]
    response = model.chat_completions(messages)
    # response = model.chat_completions(prompt)
    print(response)
