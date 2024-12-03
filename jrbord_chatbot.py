import os
import tempfile
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="JR. Board Chat", page_icon=":computer:")   

# Sidebar 의 header 및 markdown 글자 색깔 & expander 배경색 지정
st.markdown(
    """
    <style>
    /* Sidebar background and text color */
    [data-testid="stSidebar"] {background-color: #08487d;}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p {color: white;}

    /* Expander title color */
    [data-testid="stExpander"] button div[role="button"] {color: black;}
    
    /* Expander background color */
    [data-testid="stExpander"] {background-color: white;}

    /* Expander content area background color */
    [data-testid="stExpander"] > div {background-color: white;}

    /* JR. Board Chatbot 스타일 */
    .sidebar-title {
        font-size: 35px;  /* 글씨 크기 */
        font-weight: bold;  /* 굵게 */
        color: white;  /* 글자 색상 */
        text-align: center;  /* 가운데 정렬 */
        margin-bottom: 20px;  /* 아래 여백 */
    }

    /* Header 스타일: 밑줄 추가 및 글씨 크기 키움 */
    .sidebar-header {
        font-size: 25px;  /* 글씨 크기 */
        font-weight: bold;  /* 굵게 */
        color: white;  /* 글자 색상 */
        text-decoration: underline;  /* 밑줄 추가 */
        margin-top: 20px;  /* 위쪽 여백 */
        margin-bottom: 10px;  /* 아래 여백 */
    }
    
    /* 메인 헤더 스타일 */
    .main-header {
        font-size: 30px;  /* 글씨 크기 */
        font-weight: bold;  /* 굵게 */
        color: #333333;  /* 텍스트 색상 */
        display: flex;  /* 아이콘과 텍스트 정렬 */
        align-items: center;  /* 수직 중앙 정렬 */
        gap: 10px;  /* 텍스트와 아이콘 사이 간격 */
        margin-bottom: 20px;  /* 아래 여백 */
    }

    .icon {
        font-size: 26px;  /* 아이콘 크기 */
        color: #007BFF;  /* 아이콘 색상 */
    }

    /* 목록 아이템 앞 빈 삼각형 스타일 */
    .triangle-list {
        font-size: 18px;  /* 글씨 크기 */
        color: #555555;  /* 텍스트 색상 */
        margin-left: 20px;  /* 왼쪽 여백 */
        line-height: 1.6;  /* 줄 간격 */
        display: flex; /* 삼각형과 텍스트 정렬 */
        align-items: center;
    }

    /* 목록 스타일 */
    .custom-list {
        font-size: 18px;  /* 글씨 크기 */
        color: #333333;  /* 텍스트 색상 */
        margin-bottom: 15px;  /* 항목 간 간격 */
        text-decoration: underline; /* 밑줄 추가 */
    }

    /* 빈 삼각형 추가 */
    .custom-list::before {
        content: ""; /* 빈 내용 */
        display: inline-block;
        width: 0;
        height: 0;
        border-top: 6px solid transparent; /* 위쪽 투명 테두리 */
        border-bottom: 6px solid transparent; /* 아래쪽 투명 테두리 */
        border-left: 10px solid black; /* 검은색 테두리 */
        background-color: transparent; /* 삼각형 내부를 투명하게 */
        margin-right: 10px; /* 텍스트와의 간격 */
    }

    /* 사이드바 글자 스타일 */
    .sidebar-list {
        color: white;  /* 흰색 글자 */
        font-size: 16px;  /* 글씨 크기 */
        line-height: 1.8;  /* 줄 간격 */
        padding-left: 20px;  /* 왼쪽 여백 */
        list-style-type: disc; /* 목록 앞에 점 추가 */
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-header">
         KFPA 복지 기준 검색기<span>📖</span>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="custom-list">검색하고자 하는 내용을 아래 메시지 창에 입력해 주세요</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-list">입력 내용이 상세할수록 답변이 정확합니다</div>', unsafe_allow_html=True)
st.divider()

with st.sidebar:
    # JR. Board Chatbot 제목 추가
    st.markdown('<div class="sidebar-title">JR. Board Chatbot</div>', unsafe_allow_html=True)

    # About 헤더
    st.markdown('<div class="sidebar-header">About</div>', unsafe_allow_html=True)
    st.markdown("복지와 관련된 기준 기반으로 질문에 답변합니다")

    # Document list 헤더
    st.markdown('<div class="sidebar-header">Document list</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul class="sidebar-list">
            <li>가족수당지급기준</li>
            <li>국내교육연수출장비지급기준</li>
            <li>비연고지단신근무자교통보조금지급기준</li>
            <li>임직원대출운영기준</li>
            <li>자기계발지원기준</li>
            <li>자녀양육수당 지급기준</li>
            <li>중식수당 지급기준</li>
            <li>직원 차량유지비 등 지급기준</li>
            <li>직원 경조금지급기준</li>
            <li>체력단련비 지급기준</li>
            <li>출퇴근보조금지급기준</li>
            <li>피복관리기준</li>
            <li>협회가 필요로 하는 분야의 자격 인정 종목 기준</li>
        </ul>
        """,
        unsafe_allow_html=True
    )
    st.image('KV.png')
    st.image('영문시그니처(소).jpg')
    
@st.cache_resource(ttl="1h")
def get_faiss_db():
    # 임베딩 모델 설정(HuggingFace)
    model_name = "jhgan/ko-sbert-nli"
    model_kwargs= {'device' : 'cpu'}
    encode_kwargs= {'normalize_embeddings' : True}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs)
    db = FAISS.load_local("jrchat_test(1007)_4", hf, allow_dangerous_deserialization=True)
    return db

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)

class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.documents = []

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.documents = []  # 초기화

    def on_retriever_end(self, documents, **kwargs):
        self.documents = documents

def display_retrieved_documents(documents):
    with st.expander("참조 문서 확인"):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            st.markdown(f"** from {source}**")

db = get_faiss_db()
retriever = db.as_retriever(search_type="similarity",
                            search_kwargs={'k':5, 'fetch_k':8},
                            )     

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name="gpt-4o", temperature=0.1, streaming=True
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True
)

if len(msgs.messages) == 0:
    # msgs.clear()
    msgs.add_ai_message("무엇을 도와드릴까요?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
        display_retrieved_documents(retrieval_handler.documents)
