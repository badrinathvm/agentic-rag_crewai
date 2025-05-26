import sys
import warnings
import time 
import json 
from datetime import datetime
from src.agentic_rag.crew import AgenticRAG
import os
from dotenv import load_dotenv
import asyncio

warnings.filterwarnings("ignore")

async def run():
    """Run the Agentic RAG Crew"""
    #os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    
    print("\n" + "="*50)
    print("Starting CrewAI RAG System...")
    print("="*50)
    
    print("\n===== Interactive CREW Chat =====")
    print("Type 'exit' or 'quit' to end the conversation")
    print("Type 'clear' to clear conversation history")
    print("==================================\n")
    
    try:
        while True:
            query = input("Enter your query: ")
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting the program...")
                break
            
            inputs = {
                'query' : query
            }
            try:
                crew = AgenticRAG().crew()
                result = crew.kickoff(inputs = inputs)
                
                print("\n" + "="*50)
                print("CrewAI RAG System Results:")
                print("="*50)
                print(result)
                print("\n" + "="*50)
            except Exception as e:
                print("\nERROR:")
                print("-"*50)
                print(f"An error occurred while running the crew: {e}")
                print("\n" + "="*50)
                return None
    finally:
        print("\n" + "="*50)
        print("Exiting the program...")
        print("="*50)
    
if __name__ == "__main__":
    asyncio.run(run())
    