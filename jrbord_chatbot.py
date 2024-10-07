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
col1, col2 = st.columns([1, 8])

# 로고 넣기 (col1에 배치)
with col1:
    st.image("image_1.png", width=90)

# 제목 넣기 (col2에 배치)
with col2:
    st.markdown(
        """
        <div style="display: flex; align-items: center; height: 100%;">
            <h1 style='display: inline; vertical-align: middle; margin: 0;'>주니어보드 챗봇</h1>
        </div>
        """, 
        unsafe_allow_html=True
    )

# col1, col2 = st.columns([3,1])
# with col1:
#     st.title(":computer: 주니어보드 챗봇")
# with col2:
#     st.image("image.png", use_column_width=True)
    
st.subheader("복지 관련 내용 검색기")
st.markdown("- 복지관련 기준 내용을 분할하여 입력하였습니다")
st.markdown("- 검색하고자 하는 내용을 아래 메시지 창에 입력해 주세요")
st.markdown("- 입력 내용이 상세할수록 답변이 정확합니다")
st.divider()

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
                            search_kwargs={'k':3, 'fetch_k':5},
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

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
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
