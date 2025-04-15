import os
import configparser
from openai import OpenAI

# Read API key from config file
config = configparser.ConfigParser()
config_file = os.path.join(os.path.dirname(__file__), "../../config.ini")
config.read(config_file)
api_key = config['openai']['api_key']

# Create a new OpenAI client
client = OpenAI(api_key=api_key)

def prompt_model(prompt):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=[
            {"role": "user", 'content': prompt}
        ]
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    # Test the LLM client is working
    text = prompt_model("Say hello world back to me.")
    print(text)