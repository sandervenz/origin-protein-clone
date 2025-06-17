import asyncio
import json
import os
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()
api_key = os.getenv("MISTRAL_KEY")

async def get_llm_response(user_inputs):
    if not api_key:
        raise ValueError("MISTRAL_KEY is missing. Please set up your API key in .env.")

    client = Mistral(api_key=api_key)
    model = "mistral-small"

    system_message = """
    You are a helpful assistant who helps biologists generate a detailed prompt for a protein sequence generator.
    Do not ask for additional details.
    Only generate a detailed prompt.
    Output should include only the pure prompt without any additional commentary or explanation.
    Provide output in JSON format only with key "response".

    #####

    JSON format:

    {
      "response": "__YOUR_RESPONSE_HERE__"
    }

    Do not include any explanation, markdown, or extra text outside the JSON.
    The value of 'response' should contain your full answer as a string.
    """

    # **Construct message history**
    messages = [{"role": "system", "content": system_message}]
    for user_text in user_inputs:
        messages.append({"role": "user", "content": user_text})
    #print("\n"*5)
    #print(messages)
    #print("="*100)


    # Send chat completion request to Model
    try:
        response = await client.chat.stream_async(model=model, messages=messages)

        # Accumulate streamed content
        output = ""
        async for chunk in response:
            delta = chunk.data.choices[0].delta
            if delta and delta.content:
                output += delta.content  # Preserve spacing

        # Parse response into JSON
        json_output = json.loads(output)

        print(json_output)
        return json_output.get("response", output.strip())
    


    except json.JSONDecodeError:
        return "Error: Invalid response format, possibly due to rate limits."
    except Exception as e:
        return f"Error: {str(e)}"
