from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain import hub
import re

load_dotenv(override=True)

llm =ChatGoogleGenerativeAI(model="gemini-1.5-flash")

db = SQLDatabase.from_uri("mysql+mysqlconnector://root:password@localhost/books_db")
# db = SQLDatabase.from_uri("mysql+mysqlconnector://<username>:<password>@<HOST>/<databaseName>")

# print(db.get_usable_table_names())

query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

def extract_sql_query(response:str)->str:
    """Extract the sql query from model response"""
    match = re.search(r"```sql\n(.*?)\n```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()

def write_query(question:str):
    """Generate SQL query from the user question"""
    prompt = query_prompt_template.invoke({
        "dialect":db.dialect,
        "top_k":10,
        "table_info":db.get_table_info(),
        "input":question
    })
    resp = llm.invoke(prompt.to_string())
    return extract_sql_query(resp.content) 

def execute_query(query:str):
    """Execute the sql query"""
    execute_query = QuerySQLDatabaseTool(db=db)
    return execute_query.invoke(query)

def generate_response(question:str, query:str, result:str):
    """Generate response using the query result"""
    prompt = (
        "Given the following user question, corresponding to the sql query,"
        "and SQL result, answer the user question,"
        f'Question:{question}'
        f'SQL query: {query}'
        f'SQL result: {result}'
    )

    resp = llm.invoke(prompt)
    return resp.content

question = "Who is author of harry potter"
query = write_query(question)
result = execute_query(query)
answer = generate_response(question, query, result)
print(answer)