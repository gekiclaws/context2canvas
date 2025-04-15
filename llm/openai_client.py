from openai import OpenAI
client = OpenAI()

def prompt_model(prompt):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=[
            {"role": "user", 'content': prompt}
        ]
    )
    return completion.choices[0].message.content