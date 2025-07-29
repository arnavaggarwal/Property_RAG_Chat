🏡 AI Real Estate Agent

🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

1. Prerequisites

    Python 3.12.11

    An Azure account with access to Azure OpenAI Service. You will need:

        An API Key

        An Endpoint URL

        A Chat Model Deployment Name (e.g., for gpt-4)

        An Embedding Model Deployment Name (e.g., for text-embedding-ada-002)


2. Create a Virtual Environment

It's highly recommended to use a virtual environment.

For Windows, run:
Generated code

python -m venv venv
.\venv\Scripts\activate

Use code with caution.

For macOS/Linux, run:
Generated code

python3 -m venv venv
source venv/bin/activate

Use code with caution.

3. Install Dependencies

Now, install dependencies using this command:
Generated code

pip install --upgrade -r requirements.txt

Use code with caution.

4. Prepare the Data

Ensure the property data file, enhanced_property_data_with_rich_descriptions.csv, is present in the root of your project directory.

5. Run the Application

Once the setup is complete, run the Streamlit application with this command:
Generated code

streamlit run app.py

Use code with caution.

💻 How to Use

    Launch: After running the command above, a new tab should open in your web browser at http://localhost:8501.

    Enter Credentials:

        On the left sidebar, you will see input fields for your Azure OpenAI credentials.

        Enter your API Key, Endpoint, Chat Deployment Name, and Embedding Deployment Name.

        Click the "Connect" button.

    Initialization:

        The application will connect to the Qdrant vector database.

        If it's the first time running, it will create and populate the database from your CSV file. This may take a minute.

    Chat with the AI:

        Once the "Database Status" shows it is connected, you can start chatting.

        Use the chat input box at the bottom of the screen to type your requests.

    View Sources:

        For answers that involve property listings, an expandable section titled "See the X properties I considered" will appear below the AI's message. Click on it to see the details.
