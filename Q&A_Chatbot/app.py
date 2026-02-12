import os
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import ollama
from dotenv import load_dotenv
load_dotenv()

#Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot with ollama"

#Prompt Template
prompt=ChatPromptTemplate([
    ("System","You are a helpful assitant. Please respont the user queries."),
    ("user","Question:{question}")
])

def generate_Response(question,llm,temperature,max_tokens):
    llm=ollama(model=llm)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    ans=chain.invoke({'question':question})
    return ans

#Streamlit app part
# Title
st.title("Q&A Chatbot with Ollama")

#Settings Sidebar
llm=st.sidebar.selectbox("Select your llm model",["mistral","gemma3"])
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max_tokens",min_value=50,max_value=300,value=150)


#Main Page For user Input
st.write("Go ahead and ask any question:")
user_input=st.text_input("You: ")

if user_input:
    generate_Response(user_input,llm,temperature,max_tokens)
else:
    st.write("Please provide all the inputs")