#from dotenv import load_dotenv
#load_dotenv()

import os
import tiktoken
import streamlit as st
from langchain.document_loaders import ToMarkdownLoader
from langchain.text_splitter import SpacyTextSplitter
from langchain import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()

api_key = os.getenv("TOMARKDOWN_API_KEY")

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens

prompt_template = """Write a concise summary of the following:

{text}

Remember that the president of Chile is Gabriel Boric.
CONCISE SUMMARY IN SPANISH:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

st.title("TLDR en Español")

URL = st.text_input("Ingresa la página a resumir:", "")
if URL:
    loader = ToMarkdownLoader(url=URL, api_key=api_key)
    docs = loader.load()
    documents = docs[0].page_content


    num_tokens = num_tokens_from_string(documents)
    if num_tokens < 3200:
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0) 
    else:
        llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0)
	
    text_splitter = SpacyTextSplitter(chunk_size=15000)
    texts = text_splitter.create_documents([documents])

    chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=PROMPT, combine_prompt=PROMPT)
    with get_openai_callback() as cb:
        result = chain.run(texts)
        st.write(cb)
	
    st.markdown(result)
