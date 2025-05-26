from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from crewai_tools import PDFSearchTool
from src.agentic_rag.tools.custom_tool import DocumentSearchTool
from dotenv import load_dotenv
import os

pdf_tool = DocumentSearchTool(file_path='knowledge/dspy.pdf')
web_search_tool = SerperDevTool()

@CrewBase
class AgenticRAG():
    
    def __init__(self):
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
        os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
        
        self.llm = LLM(
            model="gpt-4o",
        )
        
    """Agentic RAG Crew"""
    
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    @agent
    def retriever_agent(self):
        """Retriever Agent"""
        return Agent(
            config=self.agents_config['retriever_agent'],
            verbose=True,
            tools=[web_search_tool, pdf_tool],
            llm=self.llm
        )
    
    @agent
    def response_synthesizer_agent(self):
        """Response Synthesizer Agent"""
        return Agent(
            config=self.agents_config['response_synthesizer_agent'],
            verbose=True,
            llm=self.llm
        )
        
    @task
    def retrieval_task(self):
        """Retrieval Task"""
        return Task(
            config=self.tasks_config['retrieval_task'],
            agent=self.retriever_agent(),
        )
        
    @task
    def response_task(self):
        """Response Task"""
        return Task(
            config=self.tasks_config['response_task'],
            agent=self.response_synthesizer_agent(),
        )
        
    @crew
    def crew(self):
        """Creates the AgenticRAF Crew"""
        return Crew(
            agents = self.agents,
            tasks = self.tasks,
            verbose = True
        )
            
        