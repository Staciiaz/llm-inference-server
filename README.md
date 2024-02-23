# llm-inference-server

This repository hosts an API server, utilizing [FastAPI](https://pypi.org/project/fastapi/), designed to deploy [Hugging Face models](https://huggingface.co/models) as a service, allowing users to perform inference via the [OpenAI](https://pypi.org/project/openai/) client. Please note that this repository is not officially affiliated with Hugging Face."

First, create a `.env` file following the format in `.env.example`:
```
HUGGINGFACE_TOKEN=your_huggingface_token (for gated model)
MODEL_ID=huggingface_model_id
PORT=port_to_host_the_server
```

Then, run the `main.py`:
```
python main.py
```

To see an example of how to inference the model, you can refer to the `test.py` file in this repository.

### Supported Huggingface Model
```
meta-llama/Llama-2-7b-chat-hf
google/gemma-7b-it
mistralai/Mistral-7B-Instruct-v0.2
```

### Disclaimer
This repository is maintained independently and is not officially associated with Hugging Face. Use it at your own discretion.

-----
Feel free to adjust any part of it to better fit your specific context or preferences!
