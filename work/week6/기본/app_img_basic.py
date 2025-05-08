import base64
from io import BytesIO
import streamlit as st
from PIL import Image
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
import uuid
from time import sleep
from chromadb.config import Settings
import os
import shutil

# 설정 및 상수
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "img_db"
OPENAI_API_KEY = st.secrets["OPENAI_KEY"]
DEFAULT_SYSTEM_MESSAGE = (
    "너는 이미지를 전문적으로 분석하는 AI 어시스턴트야. "
    "아래 이미지를 꼼꼼히 살펴본 뒤, 이미지 속 주요 사물·색상·구성 요소를 간결하게 요약해줘. "
    "필요하면 이미지에 있는 텍스트(간판·문구 등)도 뽑아내고, 배경·분위기도 함께 설명해줘."
)

# 초기화
model = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_KEY"])
embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_KEY"])


# 함수 정의
def images_to_docs(images: list) -> list[Document]:
    docs = []
    for image in images:
        img = Image.open(image)
        buffered = BytesIO()
        img.save(buffered, format=img.format)
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        docs.append(Document(
            page_content=f"data:image/{img.format.lower()};base64,{image_base64}",
        ))
    return docs

def get_vectorstore() -> Chroma:
    client = chromadb.EphemeralClient()
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

def format_docs(_docs):
    return [doc.page_content for doc in _docs]

def on_files_change():
    st.session_state.uploader_key += 1

# Streamlit UI

st.title("📷 이미지 기반 QA RAG 봇")
st.caption("🚀 업로드한 이미지를 기반으로 답변이 생성됩니다.")
vectordb = get_vectorstore()
# 이미지 업로드 및 벡터스토어 초기화
if images := st.file_uploader("이미지를 업로드해주세요.",type=['png', 'jpg', 'jpeg'], accept_multiple_files=True):
    with st.spinner("이미지를 벡터스토어에 추가하는 중..."):
        current_docs = images_to_docs(images)
        vectordb.add_documents(current_docs)
        
        
    for image in images:
        st.image(image)
 
    
# 메시지 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="너는 이미지를 전문적으로 분석하는 AI 어시스턴트야. 아래 이미지를 꼼꼼히 살펴본 뒤, 이미지 속 주요 사물·색상·구성 요소를 간결하게 요약해줘. 필요하면 이미지에 있는 텍스트(간판·문구 등)도 뽑아내고, 배경·분위기도 함께 설명해줘.")
    ]


# 기존 메시지 렌더링
for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("메세지를 입력해주세요."):
    
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    try:
        # 벡터스토어에서 관련 문서 검색
        msgs = []
        retriever = vectordb.as_retriever()
        docs = retriever.get_relevant_documents(prompt)
        formatted_docs = format_docs(docs)
        img_msgs = [{"role": "user", "content": []}]

        for msg in st.session_state.messages:
            if isinstance(msg, SystemMessage):
                msgs.append(msg)
            elif msg["role"] == "user":
                msgs.append(HumanMessage(
                    content=msg["content"],
                ))
            else:
                msgs.append(AIMessage(
                    content=msg["content"],
                ))

        for image_data in formatted_docs:
            img_msgs[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": image_data}
            })

        with st.chat_message("assistant"):
            result = model.invoke(msgs + img_msgs)
            st.session_state.messages.append({"role": "assistant", "content": result.content})
            response = result.content
            st.markdown(response)
    except Exception as e:
        st.error(f"질문 처리 중 오류가 발생했습니다: {e}")
        
    

