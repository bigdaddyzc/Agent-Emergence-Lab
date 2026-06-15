"""Memory system: short-term working memory + long-term semantic memory."""

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

EN_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
    "shall", "should", "may", "might", "must", "need", "dare", "ought",
    "this", "that", "these", "those", "i", "you", "he", "she", "it", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his", "its",
    "our", "their", "mine", "yours", "hers", "ours", "theirs",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "as", "until", "while", "about", "against",
    "up", "down", "and", "or", "but", "if", "so", "not",
}

# Common Chinese stopwords for filtering keyword noise
CN_STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
    "一", "个", "上", "也", "很", "到", "说", "要", "去", "你", "会",
    "着", "没", "看", "好", "自", "己", "这", "那", "对", "吗", "吧",
    "啊", "呢", "啦", "哦", "哈", "呀", "嘛", "嗯", "哈", "哈哈",
    "我们", "他们", "它们", "你们", "自己", "什么", "怎么", "这个",
    "那个", "可以", "因为", "所以", "但是", "还是", "只是", "不是",
    "如果", "虽然", "而且", "然后", "这样", "那样", "可能", "没有",
    "已经", "知道", "觉得", "真的", "就是", "不过", "一个", "那个",
    "这个", "其实", "比如", "就是", "真的", "时候", "开始", "一样",
}


@dataclass
class MemoryEntry:
    id: str
    content: str
    source_agent: str
    turn_number: int
    timestamp: float
    relevance_score: float = 0.0
    keywords: list[str] = field(default_factory=list)
    topic: str = ""
    depth_level: int = 1        # 1-5 深度层级
    ref_count: int = 0          # 被检索次数
    memory_type: str = "fact"   # insight / fact / analogy / question / metacognitive
    novelty_score: float = 0.0  # 0.0 ~ 1.0
    connections: list[str] = field(default_factory=list)  # 关联记忆ID
    is_synthetic: bool = False  # 是否由合并产生


class MemorySystem:
    """Dual-layer memory: short-term conversation buffer + long-term semantic store."""

    def __init__(self, config: dict, ollama_client=None):
        self.config = config
        self.client = ollama_client
        mem_config = config["memory"]

        self.short_term_window = mem_config.get("short_term_window", 10)
        self.extraction_interval = mem_config.get("extraction_interval", 3)
        self.retrieval_method = mem_config.get("retrieval_method", "keyword")
        self.max_memories = mem_config.get("max_memories_per_prompt", 5)
        self.consolidation_interval = mem_config.get("consolidation_interval", 5)

        # Context compression
        self.compressed_summary = ""
        self.last_compressed_turn = -1
        self.total_prompt_tokens = 0
        self.compression_threshold = mem_config.get("compression_threshold", 2800)
        self.turns_kept_verbatim = mem_config.get("turns_kept_verbatim", 3)

        # In-memory stores
        self.short_term: list[dict] = []
        self.long_term: list[MemoryEntry] = []
        self._extraction_count = 0
        self.last_retrieved: list[MemoryEntry] = []

    # ---- Short-term memory ----

    def add_turn(self, agent_name: str, content: str,
                 token_count: int, turn_number: int) -> None:
        entry = {
            "agent": agent_name,
            "content": content,
            "token_count": token_count,
            "turn_number": turn_number,
            "timestamp": time.time(),
        }
        self.short_term.append(entry)
        if len(self.short_term) > self.short_term_window:
            self.short_term.pop(0)

    def get_recent_context(self, n: Optional[int] = None) -> list[dict]:
        count = n or self.short_term_window
        return self.short_term[-count:]

    # ---- Long-term memory: extraction ----

    def extract_memories(self, text: str, source_agent: str,
                         turn_number: int, topic: str = "") -> list[MemoryEntry]:
        """Extract 1-3 key insights from text using the analyst model."""
        if not self.client:
            return []

        prompt = (
            "从以下对话中提取1-3条可以被进一步探讨的观点。\n"
            "为每条观点标注类型和新鲜度。\n\n"
            "类型可以是：insight（新洞见）、analogy（类比）、question（值得追问的问题）、"
            "metacognitive（元认知反思）、fact（观察到的事实）。\n"
            "新鲜度打分：1（很常见）- 10（全新视角）。\n\n"
            "优先提取：有新意的类比或比喻、可以往下深挖的线索或疑问、双方都认可的有趣想法、"
            "经过质疑和回应后仍然成立的结论。\n"
            "避免只说「讨论了X」「提到了Y」这类表述。\n"
            "每条不超过50字，但尽量包含足够多的关键词以便后续关联。\n\n"
            f"对话：{text}\n\n"
            "格式：类型|新鲜度|内容（每行一条）\n"
            "例如：insight|8|人类对未知的好奇心和恐怖感其实来源于同一个机制\n"
        )

        try:
            resp = self.client.generate(
                model=self.config["agents"]["agent_a"]["model"],
                prompt=prompt,
                temperature=0.3,
                max_tokens=300,
            )
        except Exception as e:
            logger.warning("Memory extraction failed: %s", e)
            return []

        lines = [l.strip() for l in resp.text.strip().split("\n") if l.strip()]
        memories = []
        for line in lines[:3]:
            # Try parsing structured format: type|novelty|content
            parts = line.split("|", 2)
            if len(parts) == 3:
                mem_type = parts[0].strip().lower()
                if mem_type not in ("insight", "analogy", "question", "metacognitive", "fact"):
                    mem_type = "fact"
                try:
                    novelty = float(parts[1].strip()) / 10.0
                except ValueError:
                    novelty = 0.0
                content = parts[2].strip()
            else:
                mem_type = self._classify_memory_type(line)
                novelty = 0.0
                content = line

            keywords = self._extract_keywords(content)
            mem = MemoryEntry(
                id=f"mem_{uuid.uuid4().hex[:8]}",
                content=content,
                source_agent=source_agent,
                turn_number=turn_number,
                timestamp=time.time(),
                keywords=keywords,
                topic=topic,
                depth_level=self.classify_depth(content),
                memory_type=mem_type,
                novelty_score=novelty,
            )
            memories.append(mem)

        return memories

    def classify_depth(self, text: str) -> int:
        """Heuristically classify content depth on 1-5 scale using keyword signals.

        Checks deepest layer first so a deep signal isn't masked by a shallower
        keyword appearing earlier in the same text (e.g. "本质上...相通" → layer 5,
        not layer 2).
        """
        text_lower = text.lower()
        # Layer 5: cross-domain synthesis
        if any(kw in text_lower for kw in ["联系到", "延伸到", "跨领域", "类比到", "相通"]):
            return 5
        # Layer 4: applications
        if any(kw in text_lower for kw in ["可以应用", "实际", "现实", "实践", "应用"]):
            return 4
        # Layer 3: principles, mechanisms
        if any(kw in text_lower for kw in ["原理", "机制", "逻辑", "核心", "因为"]):
            return 3
        # Layer 2: patterns, generalizations
        if any(kw in text_lower for kw in ["总是", "往往", "经常", "通常", "本质上"]):
            return 2
        # Layer 1: concrete examples, personal experiences
        if any(kw in text_lower for kw in ["比如", "例如", "有一次", "我见过", "我最近"]):
            return 1
        return 1

    def _classify_memory_type(self, text: str) -> str:
        """Fallback heuristic classification of memory type."""
        t = text.lower()
        if any(kw in t for kw in ["就像", "类比", "比喻", "联系到", "analogy",
                                   "similar to", "as in", "正如", "相通于"]):
            return "analogy"
        if any(kw in t for kw in ["我在想", "我发现自己", "我们是不是", "我不确定",
                                   "反思", "元认知", "wonder", "reflect", "忽略",
                                   "知识盲区", "薄弱", "最弱", "假设.*成立"]):
            return "metacognitive"
        if any(kw in t for kw in ["?", "吗", "何", "如果", "是否", "会不会"]):
            return "question"
        if any(kw in t for kw in ["新概念", "新洞见", "关键发现", "insight",
                                   "novel", "新视角", "洞见"]):
            return "insight"
        if any(kw in t for kw in ["因为", "所以", "如果", "那么", "原理",
                                   "机制", "therefore", "because"]):
            return "insight"
        return "fact"

    def should_extract(self, turn_number: int) -> bool:
        return (turn_number > 0
                and turn_number % self.extraction_interval == 0)

    def should_consolidate(self, turn_number: int) -> bool:
        return (turn_number > 0
                and (turn_number // self.extraction_interval) > 0
                and (turn_number // self.extraction_interval)
                % self.consolidation_interval == 0)

    # ---- Long-term memory: retrieval ----

    def retrieve_relevant(self, query: str, n: Optional[int] = None,
                          current_topic: str = "") -> tuple[list[MemoryEntry], dict]:
        """Return (memories, depth_stats). Tracks ref_count on retrieved memories."""
        if not self.long_term:
            self.last_retrieved = []
            return [], {"depth_distribution": {}, "total_memories": 0, "avg_depth": 0}

        count = n or self.max_memories
        query_keywords = set(self._extract_keywords(query))
        topic_keywords = set(self._extract_keywords(current_topic)) if current_topic else set()

        # Find turn range for recency boost
        max_turn = max(m.turn_number for m in self.long_term)
        min_turn = min(m.turn_number for m in self.long_term)
        turn_range = max(max_turn - min_turn, 1)

        if self.retrieval_method == "keyword":
            scored = []
            for mem in self.long_term:
                mem_keywords = set(mem.keywords)
                if not query_keywords or not mem_keywords:
                    keyword_score = 0
                else:
                    intersection = query_keywords & mem_keywords
                    union = query_keywords | mem_keywords
                    keyword_score = len(intersection) / max(len(union), 1)

                # Topic bonus: memories matching current topic get +0~0.3
                topic_score = 0
                if current_topic and mem.topic:
                    mem_topic_kw = set(self._extract_keywords(mem.topic))
                    if topic_keywords and mem_topic_kw:
                        t_intersection = topic_keywords & mem_topic_kw
                        t_union = topic_keywords | mem_topic_kw
                        topic_score = len(t_intersection) / max(len(t_union), 1) * 0.3

                # Recency boost: newest memories get up to 1.4x multiplier
                age_ratio = (mem.turn_number - min_turn) / turn_range
                recency_mult = 1.0 + 0.4 * age_ratio

                # Novelty boost: high-novelty memories get up to 1.3x multiplier
                novelty_weight = self.config.get("memory", {}).get("novelty_boost_weight", 0.3)
                novelty_mult = 1.0 + novelty_weight * mem.novelty_score

                final_score = (keyword_score + topic_score) * recency_mult * novelty_mult
                scored.append((mem, final_score))

            scored.sort(key=lambda x: x[1], reverse=True)
            self.last_retrieved = [mem for mem, s in scored[:count] if s > 0]

            # Connection tracking: auto-link co-retrieved memories
            if self.config.get("memory", {}).get("connection_tracking", True):
                for i, mem1 in enumerate(self.last_retrieved):
                    for mem2 in self.last_retrieved[i + 1:]:
                        if mem2.id not in mem1.connections:
                            mem1.connections.append(mem2.id)
                        if mem1.id not in mem2.connections:
                            mem2.connections.append(mem1.id)

        else:
            self.last_retrieved = self.long_term[:count]

        # Track ref_count and compute depth stats
        depth_counts = {}
        for mem in self.last_retrieved:
            mem.ref_count += 1
            d = mem.depth_level
            depth_counts[d] = depth_counts.get(d, 0) + 1

        total = len(self.last_retrieved)
        avg_depth = sum(d * c for d, c in depth_counts.items()) / max(total, 1) if total else 0

        depth_stats = {
            "depth_distribution": depth_counts,
            "total_memories": len(self.long_term),
            "avg_depth": round(avg_depth, 1),
        }

        return self.last_retrieved, depth_stats

    # ---- Long-term memory: consolidation ----

    def consolidate(self) -> None:
        """Phase 1: Group related memories and synthesize higher-level insights.
           Phase 2: Deduplicate (existing behavior)."""
        if len(self.long_term) < 3:
            return

        merging_enabled = self.config.get("memory", {}).get("knowledge_merging", True)
        min_group_size = self.config.get("memory", {}).get("min_group_size_for_synthesis", 3)

        if merging_enabled and self.client:
            # Phase 1: Find groups of related memories
            groups = self._find_related_groups(threshold=0.3)
            for group in groups:
                if len(group) >= min_group_size:
                    synthetic = self._synthesize_insight(group)
                    if synthetic:
                        self.long_term.append(synthetic)
                        syn_id = synthetic.id
                        for mem in group:
                            if syn_id not in mem.connections:
                                mem.connections.append(syn_id)
                        synthetic.connections = [m.id for m in group]

        # Phase 2: Deduplicate
        merged = []
        seen_contents = set()
        for mem in sorted(self.long_term, key=lambda m: len(m.content), reverse=True):
            normalized = mem.content.strip().lower().rstrip(".")
            if normalized in seen_contents:
                continue
            seen_contents.add(normalized)
            is_dup = False
            for existing in merged:
                if self._text_similarity(normalized, existing.content.lower()):
                    is_dup = True
                    break
            if not is_dup:
                merged.append(mem)

        logger.info(
            "Memory consolidation: %d -> %d entries",
            len(self.long_term), len(merged),
        )
        self.long_term = merged

    def _find_related_groups(self, threshold: float = 0.3) -> list[list[MemoryEntry]]:
        """Group memories by keyword overlap threshold. Returns groups of 2+ members."""
        groups = []
        assigned = set()
        for i, mem1 in enumerate(self.long_term):
            if mem1.id in assigned:
                continue
            cluster = [mem1]
            for mem2 in self.long_term[i + 1:]:
                if mem2.id in assigned:
                    continue
                if self._text_similarity(mem1.content, mem2.content, threshold=threshold):
                    cluster.append(mem2)
                    assigned.add(mem2.id)
            if len(cluster) > 1:
                groups.append(cluster)
            assigned.add(mem1.id)
        return groups

    def _synthesize_insight(self, group: list[MemoryEntry]) -> Optional[MemoryEntry]:
        """Use the analyst model to merge related memories into a higher-level insight."""
        texts = "\n".join(f"- {m.content}" for m in group)
        prompt = (
            "以下是几条相关的观察或见解。请将它们综合成一个更高层次的、更有概括性的洞见，"
            "不要简单地罗列或重复。\n\n"
            f"{texts}\n\n"
            "综合后的高层次洞见（一句话，不超过40字）："
        )
        try:
            resp = self.client.generate(
                model=self.config["agents"]["agent_a"]["model"],
                prompt=prompt,
                temperature=0.3,
                max_tokens=100,
            )
            content = resp.text.strip()
            if not content or len(content) < 5:
                return None
            keywords = self._extract_keywords(content)
            avg_novelty = sum(m.novelty_score for m in group) / len(group)
            return MemoryEntry(
                id=f"syn_{uuid.uuid4().hex[:8]}",
                content=content,
                source_agent=f"synthetic({', '.join(set(m.source_agent for m in group))})",
                turn_number=max(m.turn_number for m in group),
                timestamp=time.time(),
                keywords=keywords,
                topic=group[0].topic,
                depth_level=min(5, max(m.depth_level for m in group) + 1),
                memory_type="insight",
                novelty_score=min(1.0, avg_novelty + 0.1),
                is_synthetic=True,
            )
        except Exception as e:
            logger.warning("Synthesis failed: %s", e)
            return None

    def calculate_cross_connection_density(self) -> float:
        """Average connections per memory entry."""
        if not self.long_term:
            return 0.0
        total = sum(len(m.connections) for m in self.long_term)
        return total / len(self.long_term)

    def get_novel_memory_count(self, threshold: float = 0.5) -> int:
        """Count memories with novelty_score above threshold."""
        return sum(1 for m in self.long_term if m.novelty_score > threshold)

    def get_memory_type_distribution(self) -> dict[str, int]:
        """Return count of each memory type."""
        dist: dict[str, int] = {}
        for m in self.long_term:
            dist[m.memory_type] = dist.get(m.memory_type, 0) + 1
        return dist

    def get_synthetic_memory_count(self) -> int:
        """Count memories created by consolidation merging."""
        return sum(1 for m in self.long_term if m.is_synthetic)

    def update_prompt_tokens(self, token_count: int) -> None:
        """Track cumulative prompt tokens for compression decisions."""
        self.total_prompt_tokens += token_count

    def should_compress(self) -> bool:
        """Return True if prompt tokens exceed threshold and there are old turns to compress."""
        if self.total_prompt_tokens <= self.compression_threshold:
            return False
        compressible = self.short_term[:-self.turns_kept_verbatim] if len(self.short_term) > self.turns_kept_verbatim else []
        return len(compressible) >= 2

    def compress_context(self, turn_number: int) -> str:
        """Compress old turns into a summary, keeping recent turns verbatim."""
        if len(self.short_term) <= self.turns_kept_verbatim:
            return self.compressed_summary

        old_turns = self.short_term[:-self.turns_kept_verbatim]
        history_text = ""
        for entry in old_turns:
            agent = entry.get("agent", "?")
            content = entry.get("content", "")
            history_text += f"{agent}: {content[:200]}\n\n"

        if not history_text.strip():
            return self.compressed_summary

        summary = self._summarize_turns(history_text)

        # Replace old summary with re-summarized merged content to keep length bounded
        if self.compressed_summary:
            merged = f"之前的讨论：{self.compressed_summary}\n\n最近的讨论：{summary}"
            self.compressed_summary = self._summarize_turns(merged)
        else:
            self.compressed_summary = f"对话摘要：{summary}"

        self.last_compressed_turn = turn_number
        self.total_prompt_tokens = 0
        return self.compressed_summary

    def _summarize_turns(self, text: str) -> str:
        """Use the analyst model to condense turns into a 2-3 sentence summary."""
        if not self.client:
            return ""
        prompt = (
            "用两三句话概括下面这段对话的主要内容。只提炼核心论点和关键结论，不要细节。\n\n"
            f"{text}\n\n"
            "概括："
        )
        try:
            resp = self.client.generate(
                model=self.config["agents"]["agent_a"]["model"],
                prompt=prompt,
                temperature=0.3,
                max_tokens=200,
            )
            return resp.text.strip()
        except Exception as e:
            logger.warning("Context compression failed: %s", e)
            return "(压缩失败)"

    def get_compressed_context(self) -> tuple[str, list[dict]]:
        """Return (compressed_summary, recent_turns_verbatim)."""
        recent = self.short_term[-self.turns_kept_verbatim:] if self.short_term else []
        return self.compressed_summary, recent

    # ---- Persistence ----

    def save_snapshot(self, filepath: str) -> None:
        data = []
        for mem in self.long_term:
            item = {
                "id": mem.id,
                "content": mem.content,
                "source_agent": mem.source_agent,
                "turn_number": mem.turn_number,
                "timestamp": mem.timestamp,
                "keywords": mem.keywords,
                "topic": mem.topic,
                "depth_level": mem.depth_level,
                "ref_count": mem.ref_count,
                "memory_type": mem.memory_type,
                "novelty_score": mem.novelty_score,
                "connections": mem.connections,
                "is_synthetic": mem.is_synthetic,
            }
            data.append(item)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_snapshot(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.long_term = [
            MemoryEntry(
                id=item["id"],
                content=item["content"],
                source_agent=item.get("source_agent", ""),
                turn_number=item.get("turn_number", 0),
                timestamp=item.get("timestamp", 0),
                keywords=item.get("keywords", []),
                topic=item.get("topic", ""),
                depth_level=item.get("depth_level", 1),
                ref_count=item.get("ref_count", 0),
                memory_type=item.get("memory_type", "fact"),
                novelty_score=item.get("novelty_score", 0.0),
                connections=item.get("connections", []),
                is_synthetic=item.get("is_synthetic", False),
            )
            for item in data
        ]

    # ---- Internal helpers ----

    def _extract_keywords(self, text: str) -> list[str]:
        text = text.lower()
        keywords = []

        # English tokens
        eng_tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
        eng_keywords = [t for t in eng_tokens if t not in EN_STOPWORDS and len(t) > 2]
        keywords.extend(eng_keywords)

        # Chinese: jieba word segmentation
        chn_text = re.sub(r'[a-zA-Z0-9\s\'\"，。！？、；：""''（）【】《》…—\-\n]+', '', text)
        if chn_text.strip():
            import jieba
            tokens = jieba.lcut(chn_text)
            chn_keywords = [
                t for t in tokens
                if t not in CN_STOPWORDS and len(t) >= 2 and not t.isspace()
            ]
            keywords.extend(chn_keywords)

        # Deduplicate preserving order
        seen = set()
        unique = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)
        return unique

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        dot = float(np.dot(a_arr, b_arr))
        norm_a = float(np.linalg.norm(a_arr))
        norm_b = float(np.linalg.norm(b_arr))
        denom = norm_a * norm_b
        if denom == 0:
            return 0.0
        return dot / denom

    def _text_similarity(self, a: str, b: str, threshold: float = 0.25) -> bool:
        keywords_a = set(self._extract_keywords(a))
        keywords_b = set(self._extract_keywords(b))
        if not keywords_a or not keywords_b:
            return False
        jaccard = len(keywords_a & keywords_b) / len(keywords_a | keywords_b)
        # Short texts (few keywords) need lower bar to form groups
        if len(keywords_a | keywords_b) <= 5:
            return jaccard > threshold * 0.8
        return jaccard > threshold
