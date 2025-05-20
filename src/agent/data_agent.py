from typing import List, Dict, Any
import pandas as pd
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI
from langchain.tools import BaseTool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify OpenAI API key is present
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please add it to your .env file.")

class DataAnalysisTool(BaseTool):
    name: str = "data_analysis"
    description: str = "Analyzes data and provides insights based on the query"
    
    def _run(self, query: str, data: pd.DataFrame) -> str:
        """Run the data analysis tool."""
        try:
            # Basic statistical analysis
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
            stats = data[numeric_cols].describe()
            
            # Correlation analysis
            correlations = data[numeric_cols].corr()
            
            # Basic insights
            insights = []
            for col in numeric_cols:
                mean_val = data[col].mean()
                std_val = data[col].std()
                insights.append(f"{col}: Mean={mean_val:.2f}, Std={std_val:.2f}")
            
            return f"Analysis Results:\n{stats}\n\nCorrelations:\n{correlations}\n\nInsights:\n" + "\n".join(insights)
        except Exception as e:
            return f"Error in analysis: {str(e)}"

class DataAgent:
    def __init__(self):
        self.llm = OpenAI(temperature=0)
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.tools = [
            DataAnalysisTool(),
            Tool(
                name="data_summary",
                func=lambda x: str(pd.read_csv(x).describe()),
                description="Provides a summary of the dataset"
            )
        ]
        
        self.agent = self._create_agent()
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent with tools and memory."""
        prompt = StringPromptTemplate.from_template(
            """You are a data analysis assistant. Use the following tools to help analyze data:
            
            {tools}
            
            Use the following format:
            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question
            
            Begin!
            
            Question: {input}
            {agent_scratchpad}"""
        )
        
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            allowed_tools=[tool.name for tool in self.tools],
            stop=["\nObservation:"],
            memory=self.memory
        )
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )
    
    def analyze(self, query: str, data_path: str = None) -> str:
        """Analyze data based on the user's query."""
        try:
            if data_path:
                data = pd.read_csv(data_path)
                # Add data to the agent's context
                self.agent.memory.chat_memory.add_user_message(f"Data loaded from {data_path}")
            
            response = self.agent.run(query)
            return response
        except Exception as e:
            return f"Error in analysis: {str(e)}"
    
    def get_memory(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        return self.memory.chat_memory.messages 