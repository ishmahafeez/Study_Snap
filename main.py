import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv 
import string
import InstructorEmbedding
from langchain_community.document_loaders import PyPDFLoader
from InstructorEmbedding import INSTRUCTOR
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from htmlTemplates import css, user_template, bot_template
import tempfile
import os
# from langchain.chat_models import ChatOpenAI

def filter_text(text): #filter out unreadable tezt
    printable = set(string.printable)
    filtered_text = ''.join(filter(lambda x: x in printable, text))
    return filtered_text



def get_chunks(pdf_docs):
    splits = []
    for pdf_doc in pdf_docs:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_doc.read())  
            temp_file.close()  

            loader = PyPDFLoader(temp_file.name)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=150
            )   

            document_chunks = text_splitter.split_documents(documents)
            splits.extend(document_chunks)

            os.unlink(temp_file.name)

    return splits

persist_directory = './chroma'

def get_vectorstore(splits):
    embedding = OpenAIEmbeddings()
    vectordb=Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
    
)
    return vectordb

def get_conversation_chain(vectordb):
    llm=ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0)
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )


    retriever= vectordb.as_retriever()
    qa=ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
    
)
    return qa

def handle_userinput(user_question,qa):
    result=qa.invoke({"question":user_question})
    # st.write(result["answer"]) 
    
    st.write(user_template.replace("{{MSG}}",user_question),unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}",result["answer"]),unsafe_allow_html=True)
    
    



def main():
    
    load_dotenv()
    st.set_page_config(page_title="Study Snap", page_icon="📚")
    st.write(css,unsafe_allow_html=True) 
    
    if "conversation" not in st.session_state:
        st.session_state.conversation=None
    
    
    st.header("Study Snap 📝")
    user_question=st.text_input("Enter a question about your documents: ")
    print(user_question)
    
    
    
    if user_question:
        if st.session_state.conversation is not None:  # Checking if conversation chain is initialized
            handle_userinput(user_question, st.session_state.conversation)
    
    
        
    with st.sidebar:
        st.subheader("Upload your documents")
        pdf_docs = st.file_uploader("Upload PDFs here and Click Enter", accept_multiple_files=True)
        
        if st.button("Enter"):
            with st.spinner("Processing your request"):
               
                text_chunks = get_chunks(pdf_docs)
                #create vector store
                # st.write(text_chunks)
                vectorstore=get_vectorstore(text_chunks)
                st.session_state.conversation=get_conversation_chain(vectorstore)
    
    
   
    #get text chunks    
    
    
    print("vectorstore created")

    #allows to generate new msgs in convo
    
    
    
   

    #print(filtered_text)
    ##text_chunks = get_chunks(filtered_text)
    # print(text_chunks)
    # user_question=input("Enter your question: ")
    # if user_question:
    #     handle_userinput(user_question)
if __name__ == "__main__":
    main()