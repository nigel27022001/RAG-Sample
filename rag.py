import dotenv

dotenv.load_dotenv(".env")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
loader = PyPDFLoader("Documents/Retrieval-Augmented Generation for Large Language Models.pdf")
docs = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(docs)

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

vectorstore = Chroma.from_documents(documents=all_splits, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

def draft_rag_reply(question):
    response = rag_chain.invoke({"input": question})
    answer = response["answer"]
    unique_sources = set()
    for details in response["context"]:
        unique_sources.add(details.metadata["source"])
    source_str = ""
    for source in unique_sources:
        source_str += str(source)
    return answer + "\nSources:\n" + str(source_str) + "\n"

from langchain_core.messages import HumanMessage,SystemMessage

def draft_vanilla_reply(question, context=None):
    if context:
        messages = [SystemMessage(content=context), HumanMessage(content=question)]
        response = llm.invoke(messages)
    else:
        messages = [HumanMessage(content=question)]
        response = llm.invoke(messages)
    return response.content

RAG_VANILLA_CONTEXT = "RAG is Retrieval Augmented Generation in this case"
NO_CONTEXT = None
QUESTION_ONE = "What are the categories of RAG"
QUESTION_TWO = "What are the augmentation processes in RAG"

vanilla_test_cases = [("Outputs/vanilla_question_one_no_context.txt", QUESTION_ONE, NO_CONTEXT), ("Outputs/vanilla_question_two_no_context.txt", QUESTION_TWO, NO_CONTEXT), 
                    ("Outputs/vanilla_question_one_context.txt", QUESTION_ONE, RAG_VANILLA_CONTEXT), ("Outputs/vanilla_question_two_context.txt", QUESTION_TWO, RAG_VANILLA_CONTEXT)]

for index in range(len(vanilla_test_cases)):
    path, question, context = vanilla_test_cases[index]
    text_file = open(path, "w")
    text_file.write(draft_vanilla_reply(question, context))
    text_file.close()

rag_test_cases = [("Outputs/rag_question_one.txt", QUESTION_ONE), ("Outputs/rag_question_two.txt", QUESTION_TWO)]

for index in range(len(rag_test_cases)):
    path, question = rag_test_cases[index]
    text_file = open(path, "w")
    text_file.write(draft_rag_reply(question))
    text_file.close()