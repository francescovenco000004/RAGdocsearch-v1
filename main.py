# Import necessary libraries
from pinecone import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import os
import json
from flask import Flask, Response, request, jsonify
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import openai
from openai import OpenAI


app = Flask(__name__)

# Set environment variables for API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or "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"
os.environ['PINECONE_API_KEY'] = 'c3309951-90dc-4de7-b68e-2d0db12d9247'

# Initialize Pinecone connection
api_key = os.getenv("PINECONE_API_KEY") or "c3309951-90dc-4de7-b68e-2d0db12d9247"
pc = Pinecone(api_key=api_key)

# Initialize embeddings and vector store
embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")
index_name = "ai-search"
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace="ns1")

@app.route('/api/v1.0/query', methods=['POST'])
def query_endpoint():
    try:
        # Get the query from the request
        data = request.json
        query = data.get('query')
        if not query:
            return jsonify({'error': 'No query provided'}), 400

        # Perform the similarity search with k=5
        results = vectorstore.similarity_search(query, k=5)

        # Prepare results in JSON format
        results_json = []
        for i, doc in enumerate(results):
            result_entry = {
                "chunk_number": i + 1,
                "text": doc.page_content,
                "metadata": doc.metadata if doc.metadata else {}
            }
            results_json.append(result_entry)

        # Example definition of augment_prompt function
        def augment_prompt(query: str):
            # Perform the similarity search again for generating augmented prompt
            results = vectorstore.similarity_search(query, k=3)
            source_knowledge = "\n".join([x.page_content for x in results])
            augmented_prompt = f"""Using the contexts below, answer the query.

            Contexts:
            {source_knowledge}

            Query: {query}"""
            return augmented_prompt

        # Generate the augmented prompt
        augmented_prompt = augment_prompt(query)

        client = OpenAI()
        system = [{"role": "system", "content": "You are HappyBot."}]
        chat_history = []  # past user and assistant turns for AI memory
        user = [{"role": "user", "content": augmented_prompt}]
        chat_completion = client.chat.completions.create(
            messages=system + chat_history + user,
            model="gpt-3.5-turbo",
            max_tokens=1000,
            top_p=0.9,
        )

        # Prepare chat completion result
        chat_completion_json = {
            "response": chat_completion.choices[0].message.content
        }

        # Combine results and chat completion into a final JSON structure
        final_output = {
            "search_results": results_json,
            "augmented_prompt": augmented_prompt,
            "chat_completion": chat_completion_json
        }

        # Return final output as JSON
        return jsonify(final_output)

    except Exception as e:
        # Handle unexpected errors
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # for deployment
    # to make it work for both production and development
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
