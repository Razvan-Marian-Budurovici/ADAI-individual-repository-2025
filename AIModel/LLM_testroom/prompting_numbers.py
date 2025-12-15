from google import genai

with open("apiKeys.txt", "r") as file:
    apiKey = file.read().strip()

client = genai.Client(api_key=apiKey)

product = -187

response = client.models.generate_content_stream(
    model="gemini-2.5-flash", 
    contents=[f"{product} this is the number prompt test, say the numbers back to me, and tell me a random fact about it."]
)

for chunk in response:
    print(chunk.text, end="")