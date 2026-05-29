from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    messages=[
        {"role": "user", "content": "Say hello in one short sentence."}
    ],
    temperature=0,
    max_tokens=100,
)

print(response.choices[0].message.content)