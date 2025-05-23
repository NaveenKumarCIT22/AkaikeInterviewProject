from langchain_groq import ChatGroq
import os
import getpass
from langchain_core.prompts import PromptTemplate
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS


st.set_page_config(page_title="ACKO Health Insurance Help Bot")

vector_store = None
llm = None
prompt_temp = None


# @st.cache_data
# def init():
# global vector_store, llm, prompt_temp
pdf_file_path = "../data/Acko-Group-Health-Insurance.pdf"
loader = PyPDFLoader(pdf_file_path)
docs = loader.load()
print("Loaded pdf document")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Loaded Embedding Model")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)
print("Processed documents")

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

vector_store.add_documents(split_docs)
print("Added documents to vector store")

prompt = """## SYSTEM:
You are a helpful assistant that can help users by answering questions and providing useful information. You will be given context required to answer the question. The user will ask you a question and you will answer it based on the context provided. If the context does not provide enough information to answer the question, you will say "I don't know". You will not make up any information or provide any opinions. You will only provide information that is present in the context.
## SAMPLES:
### Example 1:
user: Explain the insurance policy related to loss of pay because of hospitalization
asssistant: If an Insured Person suffers an Involuntary Unemployment due to an Illness or Injury, as specified in the Schedule / Certificate of Insurance, suffered or contracted during the Coverage Period, resulting in
loss of Income, then We will pay the monthly amount specified in the Policy Schedule / Certificate of Insurance against this Benefit, for the duration of such Unemployment, up to the number of months /
days specified in the Policy Schedule / Certificate of Insurance from the date of such Involuntary Unemployment.

### Example 2:
user: What insurace policies are present related to child education
assistant: We will pay the amount specified in the Policy Schedule / Certificate of Insurance at the frequency specified in the Policy Schedule / Certificate of Insurance in respect of each surviving Dependent Child,
irrespective of whether the child is an Insured Person under this Policy 

### Example 3:
user: I met with an accident and have to return to my place of residence, what are my options
assistant: If an Insured Person suffers an Injury due to an Accident that occurs during the Coverage Period and that Injury solely and directly requires the Insured Person to return to his place of residence, then We
will reimburse the amount incurred on tickets on a Common Carrier for the Insured Person’s travel back to his place of residence with addition or modification necessitated in the Common Carrier due to such
Illness/Injury and provided to the Insured Person, up to the limit specified in the Policy Schedule / Certificate of Insurance. This Benefit will be payable provided thathe Medical Practitioner treating the Insured Person certifies in
writing that the Insured Person is suffering from the Injury in respect of which the claim is being made.

### Example 4:
user: Unexpectedly my trip got delayed. Will I get a refund.
assistant: We will pay the amount specified in the Policy Schedule / Certificate of Insurance, if an Insured Person’s journey on a Common Carrier is delayed beyond the number of hours specified in the Policy Schedule
/ Certificate of Insurance of its scheduled departure or scheduled arrival time, during the Travel Period.

### Example 5:
user: I am denied accomodation at a hotel. Is there a policy that helps me on this?
assistant: We will reimburse the cost upto the limit specified in the Policy Schedule / Certificate of Insurance of alternative accommodation required by the Insured Person due to any cancellation of the Insured
Person’s booked and confirmed accommodation by a hotel or any other provider of accommodation.

## CONTEXT: 
{context}
## USER:
{question}
## ASSISTANT:
"""
prompt_temp = PromptTemplate.from_template(prompt)

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass(
        "Enter your Groq API key: ")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    max_retries=3
)

print("LLM Initialized")


def retriever(question):
    context_docs = vector_store.similarity_search(question, k=3)
    context = "\n-----\n".join([f"{doc.page_content}" for doc in context_docs])
    print("> Context Obtained")
    return context


def get_answer(question):
    context = retriever(question)
    prmpt = prompt_temp.invoke({"context": context, "question": question})
    print("> Answer Obtained")
    return llm.invoke(prmpt)


# init()
st.title("ACKO Help")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask any doubts regarding ACKO Health Insurance"):
    print("< Query:", user_query)
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    response = get_answer(user_query).content
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append(
        {"role": "assistant", "content": response})
