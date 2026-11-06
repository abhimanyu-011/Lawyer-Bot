from groq import Groq

client = Groq(api_key="")

question = input("Ask LawyerBot: ")

prompt = f"""
You are an Indian legal assistant.

Answer the user's question clearly and accurately.

Question:
{question}
"""

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are an Indian legal assistant."},
        {"role": "user", "content": prompt}
    ]
)

print("\nLawyerBot:\n")
print(response.choices[0].message.content)