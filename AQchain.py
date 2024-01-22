import os
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from Retrival import create_qa_model

def create_qa_chain(retriever):
    # Create an instance of OpenAI language model
    llm = OpenAI()

    # Create a RetrievalQA instance with the specified parameters
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )

    return qa