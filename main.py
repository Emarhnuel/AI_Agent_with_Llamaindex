from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import ReActAgent
from llama_index.llms import OpenAI
from pdf import United_kingdom_engine

load_dotenv()


population_path = os.path.join("data", "population.csv")
population_df = pd.read_csv(population_path)

population_query_engine = PandasQueryEngine(
    df=population_df, verbose=True, instruction_str=instruction_str
)

population_query_engine.update_prompts({"pandas_prompt": new_prompt})
population_query_engine.query("what is the population of united kingdom")
 
# with open("data/United_Kingdom.pdf", "rb") as f:
#      pdf_reader = PyPDF2.PdfReader(f)
#      num_pages = len(pdf_reader.pages) 

#      pdf_text = ""
#      for page_num in range(num_pages):
#          page = pdf_reader.pages[page_num]  
#          page_text = page.extract_text()
#          pdf_text += page_text # You'll need a PDF parsing library
 
# documents = [Document(text) for text in pdf_text.split('\n')] # Adjust if needed

# UnitedKingdom_engine = GPTVectorStoreIndex.from_documents(documents)


tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine,
        metadata=ToolMetadata(
            name="population_data",
            description="this gives information at the world population and demographics",
        ),
    ),
    QueryEngineTool(
        query_engine=United_kingdom_engine,
        metadata=ToolMetadata(
            name="United_Kingdom_data",
            description="this gives detailed information about United Kingdom the country",
        ),
    ),
]

llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)
