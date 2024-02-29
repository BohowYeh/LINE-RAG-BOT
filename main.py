from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent,
    TextMessage,
    TextSendMessage,
)
from langchain.embeddings import OpenAIEmbeddings #type: ignore
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import (
        RecursiveCharacterTextSplitter)
from langchain.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import PyPDFLoader
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import RetrievalQA
import os

api = LineBotApi(os.getenv('LINE_TOKEN'))
handler = WebhookHandler(os.getenv('LINE_SECRET'))
app = Flask(__name__)

embeddings = OpenAIEmbeddings()
model = ChatOpenAI(temperature=0)
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=200,chunk_overlap=10)
index_creator = VectorstoreIndexCreator(
  embedding=embeddings,
  text_splitter=text_splitter,
  vectorstore_cls=Chroma,
  vectorstore_kwargs={
  "persist_directory":"vector"})

loaders=[]
allFileList = os.listdir('產品資料')######
for file in allFileList:
  load=PyPDFLoader(f'產品資料/{file}')
  loaders.append(load)

docsearch = index_creator.from_loaders(loaders)
docsearch = docsearch.vectorstore.persist()

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "你是專業客服人員,請根據上下文來回答問題,"
        "你不知道答案就說你不知道, 不要試圖編造答案。\n"
        "---------\n"
        "{context}\n"
        "---------\n"
        "{history}"
        ),
    # context 和 question 為 RetrievalQA 的固定用法
    HumanMessagePromptTemplate.from_template(
        "使用繁體中文回答,對問題會盡力回答,"
        "回答有關規格的問題要註明規格可能因地區和配置的不同而有所變化。:\n"
        "Q:{question}")
])

chain_type_kwargs = {"prompt":prompt,
   "memory":ConversationSummaryBufferMemory(
   llm=model,
   max_token_limit=1500,
   memory_key="history",
   input_key="question")}

db_two=Chroma(persist_directory='vector',
  embedding_function=embeddings)

qa = RetrievalQA.from_chain_type(
  llm=model,
  # as_retriever() 方法讓資料庫變成檢索器
  retriever=db_two.as_retriever(search_kwargs={"k":8}),
  chain_type_kwargs=chain_type_kwargs)

# 新增 Line Bot Webhook 處理
@app.post("/")
def callback():
  # 取得 X-Line-Signature 表頭電子簽章內容
  signature = request.headers['X-Line-Signature']

  # 以文字形式取得請求內容
  body = request.get_data(as_text=True)
  app.logger.info("Request body: " + body)

  # 比對電子簽章並處理請求內容
  try:
      handler.handle(body, signature)
  except InvalidSignatureError:
      print("電子簽章錯誤, 請檢查密鑰是否正確？")
      abort(400)

  return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    response = qa.run(event.message.text)
    api.reply_message(event.reply_token,
                      TextSendMessage(text=response))

# 啟動 Flask 應用程式
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)


