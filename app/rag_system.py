import faiss
from app.llm_utils import client, client2, get_embedding, decide_tool
import numpy as np

from app.config import (
    CHAT_MODEL
)

from app.logger_config import setup_logger

logger = setup_logger()
# 轻量级上下文充分性阈值。
# FAISS IndexFlatL2 为相似度更高的向量返回更小的距离。
# 这些阈值是经验性的，应使用较小的评估集进行调优。
MIN_CONTEXT_CHUNKS = 2
CONTEXT_TOP_N_FOR_AVG = 3
CONTEXT_MAX_BEST_DISTANCE = 2.2
CONTEXT_MAX_AVG_TOP_DISTANCE = 2.4


class RAGSystem:
    def __init__(self, chunks, top_k=20, rerank_k=10):
        self.chunks = chunks
        self.top_k = top_k
        self.rerank_k = rerank_k
        self.index = None
        self.embeddings = None

        # self.chat_history = []  # 新增记忆

    # 把rag变成一个工具
    def rag_tool(self, query):
        return self.ask(query)

    def build_index(self):
        texts = [c["text"] for c in self.chunks]

        if self.embeddings is None:  # 加缓存避免重复计算消耗API
            embeddings = [get_embedding(t) for t in texts]
            self.embeddings = np.vstack(embeddings)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    # def retrieve(self, query, k=5):
    #    query_vec = get_embedding(query).reshape(1, -1)
    #    distances, indices = self.index.search(query_vec, k)
    #    return [self.chunks[i] for i in indices[0]]

    def retrieve(self, query, k=5):
        if self.index is None:
            raise RuntimeError("FAISS index has not been built.")

        if not self.chunks:
            logger.warning("[retrieve] no chunks available in RAGSystem")
            return []

        k = min(k, len(self.chunks))

        query_vec = get_embedding(query).reshape(1, -1)
        distances, indices = self.index.search(query_vec, k)

        results = []

        for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), start=1):
            idx = int(idx)

            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = dict(self.chunks[idx])
            chunk["distance"] = float(distance)
            chunk["retrieval_rank"] = rank

            results.append(chunk)

        if results:
            best_distance = min(c["distance"] for c in results)
            logger.info(
                f"[retrieve] query='{query}', returned={len(results)}, "
                f"best_distance={best_distance:.4f}"
            )
        else:
            logger.warning(f"[retrieve] query='{query}', no valid chunks returned")

        return results

    def assess_context_sufficiency(self, retrieved_chunks):
        """
        Lightweight context sufficiency check.

        Current rule:
        - enough chunks are retrieved
        - best FAISS distance is below threshold
        - average distance of top chunks is below threshold

        This is not a perfect factuality check. It is a lightweight guardrail
        to avoid treating FAISS top-k results as sufficient only because they exist.
        """
        num_chunks = len(retrieved_chunks)

        metrics = {
            "num_chunks": num_chunks,
            "min_required_chunks": MIN_CONTEXT_CHUNKS,
            "best_distance": None,
            "avg_top_distance": None,
            "max_best_distance": CONTEXT_MAX_BEST_DISTANCE,
            "max_avg_top_distance": CONTEXT_MAX_AVG_TOP_DISTANCE,
            "reason": "",
        }

        if num_chunks < MIN_CONTEXT_CHUNKS:
            metrics["reason"] = "Not enough retrieved chunks."
            logger.info(f"[context_sufficiency] {metrics}")
            return False, metrics

        distances = [
            c.get("distance")
            for c in retrieved_chunks
            if c.get("distance") is not None
        ]

        if not distances:
            metrics["reason"] = "No FAISS distance is available for retrieved chunks."
            logger.warning(f"[context_sufficiency] {metrics}")
            return False, metrics

        sorted_distances = sorted(float(d) for d in distances)
        top_distances = sorted_distances[:min(CONTEXT_TOP_N_FOR_AVG, len(sorted_distances))]

        best_distance = sorted_distances[0]
        avg_top_distance = sum(top_distances) / len(top_distances)

        metrics["best_distance"] = round(best_distance, 4)
        metrics["avg_top_distance"] = round(avg_top_distance, 4)

        context_sufficient = (
                best_distance <= CONTEXT_MAX_BEST_DISTANCE
                and avg_top_distance <= CONTEXT_MAX_AVG_TOP_DISTANCE
        )

        if context_sufficient:
            metrics["reason"] = "Retrieved chunks meet the lightweight distance-based sufficiency rule."
        else:
            metrics["reason"] = "Retrieved chunks exist, but FAISS distances are not strong enough."

        logger.info(f"[context_sufficiency] {metrics}")

        return context_sufficient, metrics

    def assess_context_relevance_with_llm(self, question, retrieved_chunks):
        """
        LLM-based relevance gate.

        FAISS distance can tell which chunks are nearest in vector space,
        but it cannot reliably tell whether the chunks directly answer the question.
        This gate checks whether the retrieved passages are actually relevant
        enough to support an answer.
        """
        if not retrieved_chunks:
            return False, {
                "llm_relevance_check": False,
                "llm_relevance_verdict": "NO",
                "llm_relevance_reason": "No chunks retrieved.",
                "llm_relevance_error": "",
            }

        preview = ""
        for i, c in enumerate(retrieved_chunks[:3], start=1):
            source = c.get("source", "unknown")
            distance = c.get("distance")
            text = c.get("text", "")

            snippet = text[:350]

            preview += (
                f"[Chunk {i}]\n"
                f"Source: {source}\n"
                f"Distance: {distance}\n"
                f"Text: {snippet}\n\n"
            )

        prompt = f"""
                    You are a strict relevance judge for a RAG system.
                
                    Your job is to decide whether the retrieved passages contain enough information to directly answer the user's question.
                
                    Question:
                    {question}
                
                    Retrieved passages:
                    {preview}
                
                    Decision rules:
                    - Reply YES if the passages contain information that directly answers the question.
                    - Reply NO if the passages are about a different topic.
                    - Reply NO if the passages only partially overlap with the question but do not answer the key point.
                    - Reply NO if the user asks about something not supported by the retrieved passages.
                    - Do not answer the user's question.
                    - Do not explain in multiple paragraphs.
                
                    Return exactly one line in this format:
                    YES - short reason
                    or
                    NO - short reason
                  """

        try:
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )

            verdict = response.choices[0].message.content.strip()
            verdict_upper = verdict.upper()

            if verdict_upper.startswith("YES"):
                is_relevant = True
            elif verdict_upper.startswith("NO"):
                is_relevant = False
            else:
                logger.warning(f"[relevance_gate] unexpected verdict format: {verdict}")
                # If the judge output is malformed, fall back to allowing generation,
                # but record the issue in trace for observability.
                return True, {
                    "llm_relevance_check": True,
                    "llm_relevance_verdict": verdict[:200],
                    "llm_relevance_reason": "Unexpected judge output format. Falling back to distance-based result.",
                    "llm_relevance_error": "unexpected_verdict_format",
                }

            logger.info(f"[relevance_gate] verdict={verdict}")

            return is_relevant, {
                "llm_relevance_check": is_relevant,
                "llm_relevance_verdict": verdict[:200],
                "llm_relevance_reason": verdict[:200],
                "llm_relevance_error": "",
            }

        except Exception as e:
            logger.warning(f"[relevance_gate] LLM relevance check failed: {e}")

            # Do not break the whole RAG answer when the judge fails.
            # Fall back to distance-based result, and expose the error in trace.
            return True, {
                "llm_relevance_check": True,
                "llm_relevance_verdict": "FALLBACK",
                "llm_relevance_reason": "LLM relevance check failed. Falling back to distance-based result.",
                "llm_relevance_error": str(e),
            }

    def rerank(self, query, chunks):
        prompt = f"""
                You are a ranking assistant.

                Query:
                {query}

                Rank the following passages from most relevant to least relevant.

                Passages:
                """

        for i, c in enumerate(chunks):
            prompt += f"\n[{i}] {c}\n"

        prompt += "\nReturn ONLY the indices in sorted order, like [2,0,1]."

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        import ast
        try:
            return ast.literal_eval(response.choices[0].message.content)
        except:
            return list(range(len(chunks)))

    # 限制历史长度，否则会爆token
    # def trim_history(self, max_turn=3):
    #    if len(self.chat_history) > max_turn * 2:
    #        self.chat_history = self.chat_history[-max_turn * 2:]

    def ask(self, question, chat_history=None):  # 这个ask，没有返回top_k_chunks
        if chat_history is None:
            chat_history = []

        retrieved = self.retrieve(question, k=self.top_k)

        # 可以加 rerank（你已经有了）
        # context = "\n".join(retrieved)

        # rerank（用text）
        texts = [c["text"] for c in retrieved]
        sorted_indices = self.rerank(question, texts)
        best_chunks = [retrieved[i] for i in sorted_indices[:self.rerank_k]]

        # 拼context（加来源！）
        context = ""
        for c in best_chunks:
            context += f"[Source: {c['source']}]\n{c['text']}\n\n"

        # 构造messages
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer based on context and coversation history."
            }
        ]

        # 加历史对话
        messages.extend(chat_history)

        # 当前问题
        messages.append({
            "role": "user",
            "content": f"{context}\n\nQuestion: {question}"
        })

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages
        )

        answer = response.choices[0].message.content

        return answer

    def ask_with_trace(self, question, chat_history=None):
        if chat_history is None:
            chat_history = []

        retrieved = self.retrieve(question, k=self.top_k)

        # rerank（用 text）
        texts = [c["text"] for c in retrieved]
        sorted_indices = self.rerank(question, texts)
        best_chunks = [retrieved[i] for i in sorted_indices[:self.rerank_k]]

        retrieved_chunks = []
        for c in best_chunks:
            retrieved_chunks.append({
                "source": c["source"],
                "text": c["text"],
                "distance": c.get("distance"),
                "retrieval_rank": c.get("retrieval_rank"),
            })

        # 最小版 context sufficiency 规则：
        # 当前先用 retrieved_chunks 数量判断，后续可以升级为相似度阈值 / rerank 分数 / Reflection 判断
        # context_sufficient = len(retrieved_chunks) >= 2
        # context_sufficient, context_metrics = self.assess_context_sufficiency(retrieved_chunks)

        distance_sufficient, context_metrics = self.assess_context_sufficiency(retrieved_chunks)

        context_relevant, relevance_metrics = self.assess_context_relevance_with_llm(
            question=question,
            retrieved_chunks=retrieved_chunks
        )

        context_metrics.update({
            "distance_gate_passed": distance_sufficient,
            "llm_relevance_check": relevance_metrics.get("llm_relevance_check"),
            "llm_relevance_verdict": relevance_metrics.get("llm_relevance_verdict"),
            "llm_relevance_reason": relevance_metrics.get("llm_relevance_reason"),
            "llm_relevance_error": relevance_metrics.get("llm_relevance_error"),
        })

        context_sufficient = distance_sufficient and context_relevant

        if context_sufficient:
            context_metrics["final_sufficiency_reason"] = (
                "Context passed both the distance gate and the LLM relevance gate."
            )
        elif not distance_sufficient:
            context_metrics["final_sufficiency_reason"] = (
                "Context failed the distance-based retrieval gate."
            )
        elif not context_relevant:
            context_metrics["final_sufficiency_reason"] = (
                "Context passed the distance gate but failed the LLM relevance gate."
            )
        else:
            context_metrics["final_sufficiency_reason"] = (
                "Context failed the lightweight sufficiency check."
            )

        if not context_sufficient:
            return {
                "answer": (
                    "当前检索到的论文片段不足以直接支持这个问题的可靠回答。"
                    "系统不会基于不相关或证据不足的片段强行推断。"
                    "建议换一种更具体的问题，或补充包含相关内容的论文。"
                ),
                "retrieved_chunks": retrieved_chunks,
                "context_sufficient": False,
                "context_metrics": context_metrics,
            }

        # 拼 context（加来源）
        context = ""
        for c in best_chunks:
            context += f"[Source: {c['source']}]\n{c['text']}\n\n"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Answer based on context and conversation history. "
                    "If the context is not enough, clearly say the evidence is insufficient."
                )
            }
        ]

        # 保留历史对话
        messages.extend(chat_history)

        messages.append({
            "role": "user",
            "content": f"{context}\n\nQuestion: {question}"
        })

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "context_sufficient": context_sufficient,
            "context_metrics": context_metrics,
        }

    def ask_with_agent(self, question):
        decision = decide_tool(question)
        print("🧠 Decision:", decision)

        if "RAG" in decision:
            print("📚 Using RAG...")
            return self.ask(question)

        else:
            print("💬 Using LLM...")
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question},
                ]
            )

            return response.choices[0].message.content

# if __name__ == '__main__':
#    docs = load_pdfs("data")  # 你的文件夹
#    chunks = process_documents(docs)
#
#    rag = RAGSystem(chunks)
#    rag.build_index()
#
#   answer = rag.ask("What are the differences between paper1 and paper2?")
#    print(answer)
