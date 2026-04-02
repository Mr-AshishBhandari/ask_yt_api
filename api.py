from langchain_huggingface import HuggingFaceEndpointEmbeddings,ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel , RunnablePassthrough , RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma

from fastapi import FastAPI
from youtube_transcript_api import YouTubeTranscriptApi
import uvicorn

from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI()

HUGGINGFACEHUB_ACCESS_TOKEN=os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN') 
tenant =os.getenv('tenant')
database=os.getenv('database')
chroma_api_key= os.getenv('chroma_api_key')


embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-mpnet-base-v2",
    task="feature-extraction",
    huggingfacehub_api_token=HUGGINGFACEHUB_ACCESS_TOKEN,
)
llm = HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct',huggingfacehub_api_token=HUGGINGFACEHUB_ACCESS_TOKEN)
model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()


def format_docs(retrieved_docs):
        return "\n".join(doc.page_content for doc in retrieved_docs)


@app.post('/ingest')
def ingest(video_id):
    try:
        ytti_api = YouTubeTranscriptApi()
        transcript = ytti_api.fetch(video_id=video_id,languages=['en'])
        formatted_transcript = " ".join(t.text for t in transcript)
    except Exception as e:
        return {'error':str(e)}
    
    splitter  = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50,)
    documents = splitter.create_documents([formatted_transcript])

    collecion_name = f"yt_transcript_{video_id}"
    vector_store = Chroma(
    collection_name=collecion_name,
    embedding_function=embeddings,
    chroma_cloud_api_key=chroma_api_key,
    tenant=tenant,
    database=database,
    )
    vector_store.add_documents(documents=documents)

    return {'message':"success"} 

@app.post('/ask')
def ask(query,video_id):
    
    prompt = PromptTemplate(template='''
    You are an AI assistant that answers questions using a provided YouTube video transcript.

    Instructions:
    - Use ONLY the information from the transcript to answer the question.
    - If the answer is not clearly present in the transcript, say: "The transcript does not provide enough information to answer this question."
    - Be concise, clear, and accurate.
    - Do not make up information or rely on outside knowledge.
    ---

    Transcript:
    {context}

    ---

    Question:
    {query}

    ---

    Answer:
                ''',input_variables=['context','query'])
    
    collecion_name = f"yt_transcript_{video_id}"
    vector_store = Chroma(
    collection_name=collecion_name,
    embedding_function=embeddings,
    chroma_cloud_api_key=chroma_api_key,
    tenant=tenant,
    database=database,
    )
    
    retriever = vector_store.as_retriever()
    
    parallel_chain = RunnableParallel({
        "context":  retriever | RunnableLambda(format_docs),
        "query":RunnablePassthrough()
    })

    sequential_chain = prompt | model  | parser
    final_chain = parallel_chain | sequential_chain

    result = final_chain.invoke(query)

    return {"result":result}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)