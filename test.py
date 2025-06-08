from openai import OpenAI

# Paste your API key here (for testing only)
client = OpenAI(api_key="")

# List of chatbot models to test
models = [
    # GPT-4 variants (replace or add your exact model names)
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.5-preview",

    # GPT-3.5-turbo variants
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-instruct",
    "gpt-3.5-turbo-1106"
]

for model_name in models:
    print(f"\n--- Calling model: {model_name} ---")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, can you introduce yourself briefly?"}
            ],
            max_tokens=100
        )
        answer = response.choices[0].message.content
        print(f"Response from {model_name}:\n{answer}")

    except Exception as e:
        print(f"Error calling {model_name}: {str(e)}")
