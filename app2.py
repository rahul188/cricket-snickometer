import ollama

def send_prompt(prompt):
    # Use ollama.chat to send the prompt
    stream = ollama.chat(
        model='llama3.1',
        messages=[{'role': 'user', 'content': prompt}],
        stream=True,
    )
    
    # Collect the response
    response_content = ""
    for chunk in stream:
        response_content += chunk['message']['content']
        print(chunk['message']['content'], end='', flush=True)
    
    return {"response": response_content}

# Example usage
prompt = "Once upon a time"
response = send_prompt(prompt)
print("\nFull response:", response)