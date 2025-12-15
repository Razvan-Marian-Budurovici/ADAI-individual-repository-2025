from google import genai

with open("apiKeys.txt", "r") as file:
    apiKey = file.read().strip()

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key=apiKey)

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Explain how LLMs work in simple terms."
)
print(response.text)
