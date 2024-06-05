# Import necessary libraries
import openai
import os

# Function to perform zero-shot inference
def chatgpt_zero_shot_inference(api_key, test_data):
    """
    Perform zero-shot inference using the OpenAI GPT model.

    Parameters:
        api_key (str): OpenAI API key.
        test_data (dict): Dictionary containing the test data.

    Returns:
        result (str): Generated text from the model.
    """
    openai.api_key = api_key

    prompt = (
        "Generate the adapted ending to fill these three aspects:\n"
        "1. Minimal Intervention: Adjust the story's ending with minimal changes needed to align it with the altered context. The rewritten edited ending should remain as close as possible to its original ending form.\n"
        "2. Narrative Insight: Understand the story structure and make changes essential for maintaining the story's coherence and thematic consistency, avoiding unnecessary alterations.\n"
        "3. Counterfactual Adaptability: Adapt the story's course in response to the counterfactual event that diverges from the original plotline.\n\n"
        f"Premise: {test_data['premise']}\n"
        f"Initial: {test_data['initial']}\n"
        f"Original Ending: {test_data['original_ending']}\n"
        f"Counterfactual: {test_data['counterfactual']}\n\n"
        "Now, generate the adapted ending:"
    )

    print(f"Prompt: {prompt}")

    try:
        response = openai.Completion.create(
            engine="davinci-002",
            prompt=prompt,
            max_tokens=5
        )
        generated_text = response.choices[0].text.strip()

        print(f"Generated text: {generated_text}")

        return generated_text
    except Exception as e:
        print(f"API call failed with error: {e}")
        return None

# Example test data
test_data = {
    "story_id": "42b12f6d-811e-4a0f-bd1f-5d7fdde74973",
    "premise": "The soccer game was tied 3 to 3 and there was a minute left to play.",
    "initial": "Julie had never scored a goal yet, but knew today would be her day.",
    "counterfactual": "Julie was eagerly watching the game in the stands.",
    "original_ending": "Ashley passed her the ball and this was chance. She kicked as hard as she could, and the ball soared into the net. Julie's first goal won the game.",
    "edited_ending": "Ashley passed the ball and this was the chance. Another teammate kicked as hard as she could, and the ball soared into the net. Julie's got to see the goal win the game.",
    "differences": ["see", "win", "to", "teammate", "got", "Another"]
}

# Set your OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

# Run the zero-shot inference
generated_text = chatgpt_zero_shot_inference(api_key, test_data)

# Display the generated text
print(f"Generated Ending: {generated_text}")
