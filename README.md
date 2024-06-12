## Speckly: A RAG Chatbot for Speckle's Developer Documentation

<p align="center">
  <img src=ai-01.jpeg width="600px" height="600px" >
  <br>
  <em>Image obtained from <a href="https://ideogram.ai">ideogram.ai</a></em>
</p>

This repository contains a Retrieval-Augmented Generation (RAG) chatbot designed to assist developers in understanding Speckle's developer documentation. The chatbot leverages natural language processing techniques to retrieve relevant information from Speckle's documentation and generate informative responses to user queries.

### Overview

As developers, we often find ourselves navigating through extensive documentation to find the information we need. This RAG chatbot aims to streamline the process by providing a conversational interface where you can ask questions related to Speckle's developer documentation, and the chatbot will retrieve and present the relevant information in a concise and understandable manner.

### Directory Structure

- `app/`: This directory contains the server code `server.py` and the client code `client.py` for running the chatbot application.
  To start the server, run

```
python app/server.py
```

Then, to start the streamlit app, open another terminal and run

```
streamlit run app/client.py
```

- `utils/`: This directory contains the building blocks and utility functions used by the chatbot, such as document retrieval, answer generation, and grading mechanisms.

### Getting Started

To get started with the RAG chatbot for Speckle's developer documentation, follow these steps:

1. Clone the repository: `git clone https://github.com/bhargobdeka/RAG-chatbot-Speckly.git`
2. Navigate to the project directory: `cd RAG-chatbot-Speckly`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Start the server: `python app/server.py`
5. Run the streamlit app: `streamlit run app/client.py`
6. Type your question in the space provided to get a response.

### Streamlit UI

This is how the app looks like!

<p align="center">
  <img src=app-01.png width="700px" height="500px" >
  <br>
</p>

### Contributing

Contributions to this project are welcome! If you're interested in collaborating or have suggestions for improvements, please feel free to open an issue or submit a pull request.

### License

This project is licensed under the [MIT License](LICENSE).
