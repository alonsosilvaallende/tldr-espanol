# from dotenv import load_dotenv
# load_dotenv()

import os
import numpy as np
import tiktoken
import streamlit as st
from langchain.document_loaders import ToMarkdownLoader
from langchain.text_splitter import SpacyTextSplitter
from langchain import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
import langchain
from langchain.chains import LLMChain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()

api_key = os.getenv("TOMARKDOWN_API_KEY")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
prompt_template = """Write a concise summary of the following:

{text}

Remember that the president of Chile is Gabriel Boric.
CONCISE SUMMARY IN SPANISH:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

st.title("TLDR en Español")
st.markdown("TLDR en Español es una herramienta que te permite resumir el texto de una página web")

URL = st.text_input("Ingresa la dirección de la página web a resumir (https://...):", "")
def my_reading_time(num_tokens):
    num_words = 0.75*num_tokens # source: https://openai.com/pricing
    reading_time = num_words/250 # source: https://biblioguias.unex.es/c.php?g=572102&p=3944889#:~:text=El%20lector%20promedio%20est%C3%A1%20entre,con%20una%20serie%20de%20ejercicios.
    reading_time = np.ceil(reading_time)
    return reading_time

if URL:
    progress_text = "Procesando. Por favor, espere."
    my_bar = st.progress(0, text=progress_text)
    loader = ToMarkdownLoader(url=URL, api_key=api_key)
    docs = loader.load()
    my_bar.progress(50, text=progress_text)
    documents = docs[0].page_content
    num_tokens = len(encoding.encode(documents))
    st.write(f":green[Tiempo de lectura del artículo original: {my_reading_time(num_tokens):.0f} minutos]")
    if num_tokens < 3800:
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0) 
        chain = LLMChain(llm=llm, prompt=PROMPT)
    elif num_tokens < 15800:
        llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0)
        chain = LLMChain(llm=llm, prompt=PROMPT)
    elif num_tokens < 55000:
        text_splitter = SpacyTextSplitter(chunk_size=15000)
        documents = text_splitter.create_documents([documents])
        chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=PROMPT, combine_prompt=PROMPT)
    else:
        st.write(f"The text is too long to be processed at this time")
    with get_openai_callback() as cb:
        result = chain.run(documents)
        my_bar.progress(100, text="Proceso terminado")
        st.write(f":green[Tiempo de lectura del resumen: {my_reading_time(cb.completion_tokens):.0f} minuto]")
        st.write(f":green[Costo del resumen: {cb.total_cost+.01:.6f} dólares]") #2MarkDown charges 1 cent per webpage
	
    st.markdown(result)
