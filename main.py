# -*- coding: utf-8 -*-
"""校园教务智能规划 RAG 系统 —— API 服务层。

架构：FastAPI（本文件） → 多源入库管线（rag/pipeline.py） → 独立向量库 Qdrant（rag/vector_store.py）
检索：bge-m3 稠密召回 + BM25 关键词召回 + RRF 融合（rag/retriever.py）
"""
import json
import logging
import os
import re
import time

import requests
from fastapi import FastAPI, File, Form, Header, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AliasChoices, BaseModel, Field
from starlette.concurrency import run_in_threadpool

from rag import llm, pipeline, retriever, summary, vector_store
from rag.config import BASE_DIR, settings
from rag.embedding import embedder

app = FastAPI(title="教务RAG规划器", description="多源数据入库 + 独立向量库 + 混合检索")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = llm.OPENAI_API_KEY
CHAT_URL = llm.CHAT_URL
DEFAULT_MODEL = llm.DEFAULT_MODEL
MAX_UPLOAD = 30 * 1024 * 1024

# ---------------------------------------------------------------- 日志
logging.raiseExceptions = False


def _make_logger() -> logging.Logger:
    lg = logging.getLogger("rag")
    lg.setLevel(logging.INFO)
    lg.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    try:
        fh = logging.FileHandler(os.path.join(BASE_DIR, "app.log"), encoding="utf-8")
        fh.setFormatter(fmt)
        lg.addHandler(fh)
    except Exception:
        pass
    try:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        lg.addHandler(sh)
    except Exception:
        pass
    return lg


log = _make_logger()


def logi(msg: str, *args):
    """任何情况下都不向外抛异常（写日志绝不能打断 SSE）。"""
    try:
        log.info(msg, *args)
    except Exception:
        pass


# ---------------------------------------------------------------- 请求模型
class ChatRequest(BaseModel):
    model_config = {"populate_by_name": True}

    message: str = Field(
        default="",
        validation_alias=AliasChoices("message", "msg", "content", "question", "query"),
    )
    user_info: dict = Field(default_factory=dict)
    rag_context: str = ""
    use_rag: bool = True
    user_id: str = ""
    top_k: int = 0


class UrlRequest(BaseModel):
    url: str = ""
    user_id: str = ""


class DeleteRequest(BaseModel):
    user_id: str = ""
    doc_id: str = ""


class TextRequest(BaseModel):
    text: str = ""
    title: str = "粘贴文本"
    user_id: str = ""


# ---------------------------------------------------------------- 提示词
def build_system_prompt(user_info: dict, rag_context: str, use_rag: bool, sources: list | None = None) -> str:
    user_info = user_info or {}
    courses = user_info.get("course_info") or {}
    course_list = courses.get("courseList") or []
    if isinstance(course_list, list) and course_list:
        finished = "、".join(
            f"{c.get('name', '')}({c.get('credit', '')}学分,{c.get('type', '')})"
            for c in course_list if isinstance(c, dict)
        )
    else:
        finished = user_info.get("finished_courses") or "暂无"

    if course_list or user_info.get("totalCredit"):
        credit_line = (
            f"已修总学分：{courses.get('totalCredit', user_info.get('totalCredit', '未填写'))}；"
            f"已修必修：{courses.get('requiredCredit', '未填写')}；"
            f"已修选修：{courses.get('electiveCredit', '未填写')}；"
            f"已完成学期数：{courses.get('finishedTerm', '未填写')}"
        )
    else:
        credit_line = "已修学分：未填写（请在“授权同步”里录入）"

    if use_rag and rag_context:
        kb_block = rag_context
        cite_rule = ("8. 上面的参考资料按 [1][2] 编号给出。凡引用知识库内容，必须在句末标注对应编号（如 [1]），"
                     "不要编造不存在的编号；知识库未覆盖的内容用通用知识补充并标注“（通用建议）”。")
        src_count = len(sources or [])
        kb_desc = f"本次检索到 {src_count} 条资料片段，编号与来源一一对应。"
    else:
        kb_block = "无（本次没有检索到可用的知识库内容）"
        cite_rule = "8. 本次没有检索到知识库资料，凡涉及具体学分数字、条款，必须注明“（通用建议，请以学院培养方案为准）”。"
        kb_desc = ""

    return f"""你是校园教务智能规划助手，专门帮本科生做学分核算、选课规划、毕业要求检查、复习备考计划。
用户基础信息：
学校：{user_info.get('school', '未填写')}
学院：{user_info.get('college', '未填写')}
专业：{user_info.get('major', '未填写')}
年级：{user_info.get('grade', '未填写')}
{credit_line}
已修课程：{finished}

要求：
1. 必须直接给出完整答案，不要反问、不要只说“需要更多信息”；缺信息时按通用培养方案给出可执行方案并注明假设。
2. 输出结构化、详细的方案：分章节、表格、要点，不要简短回答。
3. 做学分规划时，对照毕业要求，用表格列“模块/要求学分/已修/还差/风险提示”。
4. 选课规划要给出：课程名、学分、类型、推荐理由、避坑建议（难度/给分/作业量）。
5. 备考复习规划要拆分到天，给出知识点清单、复习顺序、时间配比。
6. 使用 markdown 格式输出，表格清晰。
7. 优先依据知识库原文作答；原文没有的用通用知识补充并标注。
{cite_rule}

知识库参考资料（{kb_desc}）：
{kb_block}
"""


def sse_text(text: str) -> str:
    return "data: " + json.dumps({"choices": [{"delta": {"content": text}}]}, ensure_ascii=False) + "\n\n"


def sse_event(obj: dict) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


def sse_error(msg: str) -> str:
    return sse_text(f"\n\n> ⚠️ {msg}\n")


def model_headers() -> dict:
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }


def call_model_once(messages: list, temperature: float = 0.7, timeout: int = 120) -> str:
    """非流式调用：流式兜底 + 上传/链接后的要点提取共用。"""
    return llm.call_model_once(messages, temperature=temperature, timeout=timeout)


CONTINUE_PROMPT = (
    "上面的回答被网络中断截断了。请从中断处继续输出剩余内容：不要重复已经写过的部分，"
    "不要重新开头，也不要解释，直接接着上次的最后一个字往下写。"
)


def _strip_overlap(prev: str, new: str, window: int = 300) -> str:
    """续写时模型常把上一段结尾重写一遍，这里把重叠的前缀去掉。"""
    if not prev or not new:
        return new
    limit = min(len(prev), len(new), window)
    for k in range(limit, 0, -1):
        if prev.endswith(new[:k]):
            return new[k:]
    return new


def stream_gen(user_msg: str, user_info: dict, rag_context: str, use_rag: bool,
               sources: list | None = None, rag_stats: dict | None = None):
    system_prompt = build_system_prompt(user_info, rag_context, use_rag, sources).strip()
    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    messages = list(base_messages)

    # 先把引用来源推给前端，回答结束后可以直接点开核对原文
    yield sse_event({"type": "sources", "sources": sources or [], "rag": rag_stats or {}})

    if not CHAT_URL or not OPENAI_API_KEY:
        yield sse_error("后端未配置 OPENAI_API_KEY 或 OPENAI_BASE_URL，请检查 backend/.env 后重启服务。")
        yield "data: [DONE]\n\n"
        return

    logi("请求 %s | 模型 %s | 用户 %s | 问题：%s", CHAT_URL, DEFAULT_MODEL,
         (rag_stats or {}).get("collection", "-"), user_msg[:60])

    full_text = ""
    finish_reason = None
    ok = False
    reported = False

    for attempt in (1, 2):
        try:
            resp = requests.post(
                CHAT_URL,
                headers=model_headers(),
                json={"model": DEFAULT_MODEL, "messages": messages, "stream": True, "temperature": 0.7},
                stream=True,
                timeout=(10, 180),
            )
            logi("第 %d 次流式请求，状态码 %s", attempt, resp.status_code)
            if resp.status_code != 200:
                detail = (resp.text or "")[:300]
                raise RuntimeError(f"模型接口返回 HTTP {resp.status_code}：{detail}")

            dedupe_done = (attempt == 1)
            pending = ""

            for item in llm.iter_sse_content(resp):
                if item is None:
                    break
                if isinstance(item, tuple) and item and item[0] == "__FINISH__":
                    finish_reason = item[1]
                    continue
                piece = item
                if not dedupe_done:
                    pending += piece
                    if len(pending) < 120:
                        continue
                    piece = _strip_overlap(full_text, pending)
                    pending = ""
                    dedupe_done = True
                if piece:
                    full_text += piece
                    yield sse_text(piece)

            if pending and not dedupe_done:
                piece = _strip_overlap(full_text, pending)
                if piece:
                    full_text += piece
                    yield sse_text(piece)

            ok = True
            break

        except Exception as e:
            logi("第 %d 次流式调用失败：%r", attempt, e)
            if attempt == 1:
                if full_text.strip():
                    messages = list(base_messages) + [
                        {"role": "assistant", "content": full_text[-1200:]},
                        {"role": "user", "content": CONTINUE_PROMPT},
                    ]
                continue
            if not full_text.strip():
                yield sse_error(f"模型调用失败：{e}")
                reported = True
            else:
                yield sse_text("\n\n> ⚠️ 连接中断，以上内容可能不完整；回复“继续”可接着写。")
                reported = True
            break

    if not full_text.strip() and not ok:
        try:
            text = call_model_once(base_messages)
            if text.strip():
                full_text = text
                ok = True
                for i in range(0, len(text), 60):
                    yield sse_text(text[i:i + 60])
        except Exception as e:
            logi("非流式兜底失败：%r", e)
            if not reported:
                yield sse_error(f"本次没有拿到模型回答，请重试。（原因：{e}）")
                reported = True

    if full_text.strip():
        logi("回答完成：%d 字，finish=%s", len(full_text), finish_reason)
    elif not reported:
        yield sse_error("模型没有返回任何内容，请重试，或检查 .env 里的模型名是否可用。")

    if finish_reason == "length":
        yield sse_text("\n\n> 提示：本次回答已达到模型输出长度上限，回复“继续”可接着输出。")
    if finish_reason == "content_filter":
        yield sse_text("\n\n> 提示：内容被模型安全策略拦截，请换个问法。")

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------- 对话接口
@app.post("/chat_stream")
async def chat_stream(req: ChatRequest, x_user_id: str = Header(default="")):
    user_msg = (req.message or "").strip()
    user_id = (req.user_id or x_user_id or "anonymous").strip()
    if not user_msg:
        async def empty_gen():
            yield sse_error("没有收到问题内容，请重新输入。")
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty_gen(), media_type="text/event-stream")

    if req.rag_context:
        rag_ctx, sources, rag_stats = req.rag_context, [], {"backend": "manual"}
    elif req.use_rag:
        try:
            result = await run_in_threadpool(retriever.retrieve, user_id, user_msg, req.top_k or None)
            rag_ctx, sources, rag_stats = result["context"], result["sources"], result["stats"]
        except Exception as e:
            logi("检索失败：%r", e)
            rag_ctx, sources, rag_stats = "", [], {"backend": "error", "error": str(e)[:120]}
    else:
        rag_ctx, sources, rag_stats = "", [], {"backend": "disabled"}

    gen = stream_gen(
        user_msg=user_msg,
        user_info=req.user_info or {},
        rag_context=rag_ctx,
        use_rag=req.use_rag,
        sources=sources,
        rag_stats=rag_stats,
    )
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------- 多源入库
def _ingest_one(user_id: str, filename: str, content: bytes, with_summary: bool = True) -> dict:
    result = pipeline.ingest_bytes(user_id, filename, content)
    text = result.pop("text", "")
    summary_md, mode = summary.summarize_text(text, filename) if with_summary else ("", "skipped")
    if with_summary and summary_md:
        try:
            pipeline.upsert_summary(user_id, result["doc_id"], result["title"], result["source"],
                                    summary_md, result["source_type"])
        except Exception as e:
            logi("摘要写回知识库失败：%r", e)
    result.update({
        "msg": f"《{filename}》已入库：{result['chunks']} 个片段，去重 {result['duplicates']} 个",
        "filename": filename,
        "chunk_count": result["chunks"],
        "chars": result["characters"],
        "summary": summary_md,
        "summary_mode": mode,
        "preview": result["sample"],
    })
    return result


@app.post("/upload_doc")
async def upload_doc(file: UploadFile = File(...), user_id: str = Form(""),
                     x_user_id: str = Header(default="")):
    uid = (user_id or x_user_id or "anonymous").strip()
    content = await file.read()
    if not content:
        return JSONResponse(status_code=400, content={"msg": "文件为空"})
    if len(content) > MAX_UPLOAD:
        return JSONResponse(status_code=400, content={"msg": f"文件超过 {MAX_UPLOAD // 1024 // 1024}MB，请压缩后再上传"})
    try:
        res = await run_in_threadpool(_ingest_one, uid, file.filename or "未命名", content)
        return res
    except pipeline.IngestError as e:
        return JSONResponse(status_code=400, content={"msg": str(e)})
    except Exception as e:
        logi("上传失败：%r", e)
        return JSONResponse(status_code=400, content={"msg": f"入库失败：{e}"})


@app.post("/upload_docs")
async def upload_docs(files: list[UploadFile] = File(...), user_id: str = Form(""),
                      x_user_id: str = Header(default="")):
    uid = (user_id or x_user_id or "anonymous").strip()
    results, failed = [], []
    for f in files:
        content = await f.read()
        name = f.filename or "未命名"
        if not content:
            failed.append({"filename": name, "msg": "文件为空"})
            continue
        if len(content) > MAX_UPLOAD:
            failed.append({"filename": name, "msg": "文件超过 30MB"})
            continue
        try:
            results.append(await run_in_threadpool(_ingest_one, uid, name, content))
        except Exception as e:
            failed.append({"filename": name, "msg": str(e)})
    return {
        "msg": f"成功入库 {len(results)} 个文件，失败 {len(failed)} 个",
        "results": results,
        "failed": failed,
    }


@app.post("/fetch_url")
async def fetch_url(req: UrlRequest, x_user_id: str = Header(default="")):
    url = (req.url or "").strip()
    uid = (req.user_id or x_user_id or "anonymous").strip()
    if not url.startswith(("http://", "https://")):
        return JSONResponse(status_code=400, content={"msg": "请填写以 http:// 或 https:// 开头的完整网址"})

    def _run():
        result = pipeline.ingest_url(uid, url)
        text = result.pop("text", "")
        summary_md, mode = summary.summarize_text(text, result["title"])
        try:
            pipeline.upsert_summary(uid, result["doc_id"], result["title"], url, summary_md, "web")
        except Exception as e:
            logi("网页摘要写回失败：%r", e)
        result.update({
            "msg": f"《{result['title']}》已入库：{result['chunks']} 个片段",
            "url": url,
            "chunk_count": result["chunks"],
            "chars": result["characters"],
            "summary": summary_md,
            "summary_mode": mode,
        })
        return result

    try:
        return await run_in_threadpool(_run)
    except Exception as e:
        return JSONResponse(status_code=400, content={"msg": f"网页抓取失败：{e}"})


@app.post("/ingest_text")
async def ingest_text(req: TextRequest, x_user_id: str = Header(default="")):
    uid = (req.user_id or x_user_id or "anonymous").strip()
    try:
        result = await run_in_threadpool(pipeline.ingest_text, uid, req.text, req.title)
    except Exception as e:
        return JSONResponse(status_code=400, content={"msg": str(e)})
    result.pop("text", None)
    return {"msg": f"已入库 {result['chunks']} 个片段", **result}


# ---------------------------------------------------------------- 知识库管理
@app.get("/kb/stats")
async def kb_stats(user_id: str = Query(default=""), x_user_id: str = Header(default="")):
    uid = (user_id or x_user_id or "anonymous").strip()

    def _run():
        return pipeline.stats(uid)

    try:
        return await run_in_threadpool(_run)
    except Exception as e:
        return JSONResponse(status_code=503, content={"msg": f"向量库不可用：{e}"})


@app.get("/kb/documents")
async def kb_documents(user_id: str = Query(default=""), x_user_id: str = Header(default="")):
    uid = (user_id or x_user_id or "anonymous").strip()
    try:
        docs = await run_in_threadpool(pipeline.list_documents, uid)
        return {"user_id": uid, "count": len(docs), "documents": docs}
    except Exception as e:
        return JSONResponse(status_code=503, content={"msg": f"向量库不可用：{e}"})


@app.post("/kb/delete")
async def kb_delete(req: DeleteRequest, x_user_id: str = Header(default="")):
    uid = (req.user_id or x_user_id or "anonymous").strip()
    if not req.doc_id:
        return JSONResponse(status_code=400, content={"msg": "缺少 doc_id"})
    try:
        removed = await run_in_threadpool(pipeline.delete_document, uid, req.doc_id)
        return {"msg": f"已删除 {removed} 个片段", "doc_id": req.doc_id}
    except Exception as e:
        return JSONResponse(status_code=503, content={"msg": f"删除失败：{e}"})


@app.post("/clear_user_collection")
async def clear_user_collection(user_id: str = Query(default=""), x_user_id: str = Header(default="")):
    uid = (user_id or x_user_id or "anonymous").strip()
    try:
        await run_in_threadpool(vector_store.get_store(embedder.collection, settings.embed_dim).delete_user, uid)
        return {"msg": "个人知识库已清空", "user_id": uid}
    except Exception as e:
        return JSONResponse(status_code=503, content={"msg": f"清空失败：{e}"})


@app.get("/clear_rag")
async def clear_rag():
    def _run():
        store = vector_store.get_store(embedder.collection, settings.embed_dim)
        store.drop_all()
        store.ensure()

    try:
        await run_in_threadpool(_run)
        return {"msg": "知识库已清空"}
    except Exception as e:
        return JSONResponse(status_code=503, content={"msg": f"清空失败：{e}"})


@app.get("/search")
async def search(q: str = Query(...), user_id: str = Query(default=""), top_k: int = Query(default=5),
                 x_user_id: str = Header(default="")):
    uid = (user_id or x_user_id or "anonymous").strip()
    try:
        return await run_in_threadpool(retriever.search_debug, uid, q, top_k)
    except Exception as e:
        return JSONResponse(status_code=503, content={"msg": f"检索失败：{e}"})


@app.get("/health")
async def health():
    def _run():
        db = vector_store.health()
        try:
            store = vector_store.get_store(embedder.collection, settings.embed_dim)
            points = store.count()
            backend = store.backend
            collection = store.collection
        except Exception as e:
            if db.get("reachable"):
                # 向量库在线，但集合还没建：首次入库会自动创建，不应报成不可用
                points, backend, collection = 0, "qdrant", embedder.collection
                db["error"] = "集合尚未创建，首次入库时自动创建"
            else:
                points, backend, collection = -1, "unavailable", embedder.collection
                db["error"] = str(e)[:200]
        return {
            "ok": True,
            "chat_url": CHAT_URL,
            "model": DEFAULT_MODEL,
            "has_key": bool(OPENAI_API_KEY),
            "embedding": {"mode": embedder.mode, "model": settings.embed_model, "dim": settings.embed_dim},
            "vector_db": {**db, "backend": backend, "collection": collection, "points": points},
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "retrieval": {"dense_top_k": settings.dense_top_k, "lexical_top_k": settings.lexical_top_k,
                          "final_top_k": settings.final_top_k, "rrf_k": settings.rrf_k},
        }

    return await run_in_threadpool(_run)


@app.get("/")
async def index():
    path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"msg": "backend 已启动，请访问 /login.html 或 /index.html"}


@app.get("/index.html")
async def index_page():
    """登录页跳转用的是相对路径 index.html，直接访问 /index.html 也要能打开。"""
    path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse(status_code=404, content={"msg": "index.html 不存在"})


@app.get("/login.html")
async def login_page():
    """页面里用相对路径跳转 login.html，直接访问 127.0.0.1:8000/login.html 时要能打开。"""
    path = os.path.join(BASE_DIR, "login.html")
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse(status_code=404, content={"msg": "login.html 不存在"})


# 前端依赖（marked / font-awesome）改为本地 vendor 提供，不依赖被墙的 CDN
VENDOR_DIR = os.path.join(BASE_DIR, "vendor")
if os.path.isdir(VENDOR_DIR):
    app.mount("/vendor", StaticFiles(directory=VENDOR_DIR), name="vendor")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
