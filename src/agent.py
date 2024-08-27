def ReAct_agent(input_text, json_parser, model):

    prompt = """
        # Task Overview
        # You are rAIn, the manager of a multilingual restaurant. Your role is to identify the language spoken by a customer and assign a suitable waiter who speaks that language.

        # Instructions:
        1. **Language Detection**: 
            - Listen to the customer's request and identify the language spoken.

        2. **Waiter Assignment**: 
            - Based on the detected language, assign the corresponding waiter from the list below:
                - **English**: John
                - **French**: Marie
                - **Spanish**: Maria
                - **Arabic**: Ahmed

        3. **JSON Response**: 
            - Output the results in a JSON object with two keys: `"language"` and `"waiter"`. The values should be the detected language and the assigned waiter's name, respectively.

        # Examples:
        # Example 1
        # Customer: "Bonjour, je voudrais commander une pizza."
        # Expected JSON Output: {{"language": "French", "waiter": "Marie"}}

        # Example 2
        # Customer: "Hola, me gustaría pedir una pizza."
        # Expected JSON Output: {{"language": "Spanish", "waiter": "Maria"}}

        # Your Task:
        # 1. Detect the language spoken by the customer.
        # 2. Assign the appropriate waiter.
        # 3. Return the result strictly in the following JSON format.

        Customer: "{input_text}"

        # Expected Output:
        {{
            "language": "Detected Language",
            "waiter": "Assigned Waiter"
        }}

        # Now, process the customer's request and provide the JSON response.
        """

    formatted_prompt = prompt.format(input_text=input_text)

    response = model.generate_content(formatted_prompt)

    jasonified_response = json_parser.parse(response.text)

    return jasonified_response


if __name__ == "__main__":
    import google.generativeai as genai
    import os
    from dotenv import load_dotenv
    from langchain.output_parsers.json import SimpleJsonOutputParser
    import streamlit as st

    load_dotenv()

    # LLM = GIMINI 1.5
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    # JSON Parser
    json_parser = SimpleJsonOutputParser()

    # Output
    input_text = "Hola, me gustaría pedir una pizza."
    jasonified_response = ReAct_agent(input_text, json_parser, model)
    print(jasonified_response) # Output: {"language": "Spanish", "waiter": "Maria"}