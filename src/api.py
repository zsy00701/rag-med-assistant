import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.rag import get_rag_chain

# 加载 .env 文件
load_dotenv()

# ================= 数据模型 =================
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    status: str = "success"

# ================= FastAPI 设置 =================
app = FastAPI(
    title="肝胆胰康复助手 API",
    description="基于 RAG 的智能问答服务",
    version="1.0"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
rag_chain = None

@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    global rag_chain
    try:
        if "OPENAI_API_KEY" not in os.environ:
            print("⚠️  警告: 未设置 OPENAI_API_KEY")
        
        # 加载 RAG 链
        rag_chain = get_rag_chain()
        print("✅ RAG 模型链加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
    
@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """对话接口"""
    global rag_chain
    if not rag_chain:
        raise HTTPException(status_code=500, detail="模型未初始化")
    
    try:
        # 调用 RAG 链
        answer = rag_chain.invoke(request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        print(f"推理错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)