from google import genai

with open("apiKeys.txt", "r") as file:
    apiKey = file.read().strip()

client = genai.Client(api_key=apiKey)

SYSTEM_CONTEXT = (
    "You are a helpful assistant for a door manufacturing company."
    "Answer in a concise and professional manner."
    "Your job is to explain the meanings of numbers provided to you."
    "Numbers under zero mean overstocking, numbers over 0 mean understocking."
)

config = genai.types.GenerateContentConfig(
    system_instruction=SYSTEM_CONTEXT,
    max_output_tokens=500
)

prompt = "who are you? also: -35 item_1, 14 item_2, -86 item_3"

response = client.models.generate_content(
    model="gemini-2.5-flash",
    config=config,
    contents=prompt,
)

print(response.text)