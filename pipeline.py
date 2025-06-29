# Auszüge aus dem RHvB Code für eine RAG Pipeline

#%% Pakete initalisieren
from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import  SentenceSplitter
from llama_index.readers.docling import DoclingReader
from llama_index.readers.file.flat.base import FlatReader
import chromadb
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever, AutoMergingRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import QueryBundle
import os

#%% Variablen definieren
fpath = 'Rechnungshofberichte/Berlin/Jahresberichte'
chromapath = 'Vektordatenbank'
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("OPENAI_API_KEY")
api_version = os.getenv("OPENAI_API_VERSION")

#%% Embedding Modell und LLM initialisieren
embedding_model = AzureOpenAIEmbedding(
            model="text-embedding-3-small",
            azure_endpoint = azure_endpoint,
            api_key = api_key,
            api_version = api_version,
        )

llm = AzureOpenAI(
            model="gpt-4o-mini",
            deployment_name = "gpt-4o-mini",
            azure_endpoint = azure_endpoint,
            api_key = api_key,
            api_version=api_version
        )

#%% Dokumente laden und Index erstellen
#Dokumente laden
dir_reader = SimpleDirectoryReader(
    input_dir=fpath,
    file_extractor = {'.md': FlatReader()},
    recursive=True
)
documents=dir_reader.load_data()  

#Chunks erstellen
parsing_method = 'SentenceSplitter'
if parsing_method == 'SentenceSplitter':
    node_parser = SentenceSplitter(
        chunk_size=1024, 
        chunk_overlap = 30,
        )   
nodes = node_parser.get_nodes_from_documents(documents) 

#ChromaDB initialisieren
chroma_client = chromadb.PersistentClient(path=chromapath)
chroma_collection = chroma_client.get_or_create_collection("Jahresberichte")
print(f"START: Chroma Collection hat {chroma_collection.count()} Vektoren.")

vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection,
    mode="append"
    )

storage_context = StorageContext.from_defaults(vector_store=vector_store)

#Zusätzliches Preprocessing durchführen (Metadaten, Zusammenfassung, ..)
transformations = [
    node_parser,
]

#Index erstellen
index = VectorStoreIndex.from_documents(
            documents=documents, 
            transformations=transformations,
            storage_context=storage_context,
            embed_model=embedding_model,
            show_progress=True,
        )
index.storage_context.persist(persist_dir=chromapath)
print(f"ENDE: Chroma hat nun {chroma_collection.count()} Vektoren.")

#%% Promt Templates
qa_prompt_str = (
    "Das ist die Datengrundlage:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Beantworte die folgende Frage nur anhand der Datengrundlage: "
    "{query_str}\n"
)

refine_prompt_str = (
    "Falls möglich, verbessere die Antwort anhand dieser Datengrundlage:\n"
    "------------\n"
    "{query_str}\n "
    "Falls die Antwort nicht hilfreich ist, verwende wieder die ursprüngliche Antwort.\n"
    "ursprüngliche Antwort: {existing_answer}"
)

#Chat templates definieren
chat_text_qa_msgs = [
    ChatMessage(role=MessageRole.SYSTEM, content="Du bist ein KI-Assistent der Berliner Verwaltung, der auf Basis einer Datengrundlage sinnvolle Antworten generiert.\n" 
                "Beachte die gegebene Datengrundlage, fokussiere dich auf relevante Inhalte und verändere NIEMALS Fakten, Namen, Berufsbezeichnungen, Zahlen oder Datumsangaben."),
    ChatMessage(role=MessageRole.USER, content=qa_prompt_str),
]
text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)

chat_refine_msgs = [
    ChatMessage(role=MessageRole.SYSTEM, content="Du bist ein KI-Assistent der Berliner Verwaltung, der auf Basis einer Datengrundlage sinnvolle Antworten generiert.\n" 
                "Beachte die gegebene Datengrundlage, fokussiere dich auf relevante Inhalte und verändere NIEMALS Fakten, Namen, Berufsbezeichnungen, Zahlen oder Datumsangaben."),
    ChatMessage(role=MessageRole.USER, content=refine_prompt_str),
]

refine_template = ChatPromptTemplate(chat_refine_msgs)

#%% Query Engine initialisieren
def custom_query_engine(
        index,
        text_qa_template=None,
        llm=None,
        ):

    response_synthesizer = get_response_synthesizer(
        text_qa_template=text_qa_template,
        llm=llm, 
        response_mode="compact" 
        )
    
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k= 5,
        )
    
    query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
  
    return query_engine

# %% Frage stellen
user_input = 'Welche Aussage trifft der Rechnungshof von Berlin zur Finanzlage im Jahr 2024?'
response = custom_query_engine(index, text_qa_template, llm).query(user_input)
print(response.response)

#%% Ideen:

'''
https://docs.llamaindex.ai/en/stable/optimizing/production_rag/

- Decoupling chunks used for retrieval vs. chunks used for synthesis
    1. Embed a document summary, which links to chunks associated with the document.
    2. Embed a sentence, which then links to a window around the sentence.

- Structured Retrieval for Larger Document Sets
    1. Metadata Filters + Auto Retrieval
    2. Store Document Hierarchies (summaries -> raw chunks) + Recursive Retrieval 

- Dynamically Retrieve Chunks Depending on your Task
    .. eventuell in Zukunft ..












'''