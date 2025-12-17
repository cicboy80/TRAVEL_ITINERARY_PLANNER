import os
import numpy as np
from typing import List, Dict, Any, Optional
from openai import OpenAI
from crewai.tools import BaseTool
from dotenv import load_dotenv

load_dotenv()
os.environ["CHROMA_OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class SemanticRankingTool(BaseTool):
    """
    CrewAI-compatible tool that semantically ranks and filters candidate places
    based on user preferences, ratings, popularity, and optional distance.
    """

    name: str = "Semantic Ranking Tool"
    description: str = (
        "Ranks restaurants and activities using semantic similarity between user preferences "
        "and place descriptions or reviews, factoring in ratings, popularity, and proximity."
    )

    def _run(
        self,
        user_preferences: Any,
        candidates: Any,
        top_k: int = 5,
        distance_weight: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Args:
            user_preferences: Natural language description of preferences (e.g. 'romantic local food')
            candidates: List of dicts containing place details
            top_k: Number of top results to return
            distance_weight: Weight applied to penalize distant places (0–1)
        """

        # 🧹 Sanitize inputs in case CrewAI passes metadata dicts
        if isinstance(user_preferences, dict):
            user_preferences = user_preferences.get("description") or user_preferences.get("value") or str(user_preferences)
        user_preferences = str(user_preferences).strip()

        if isinstance(candidates, dict):
            candidates = candidates.get("candidates") or candidates.get("items") or candidates.get("results") or candidates

        if not isinstance(candidates, list):
            raise ValueError("`candidates` must be a list of place dictionaries.")

        if not candidates:
            return []

        # ✅ Build text representations of all candidates
        texts = []
        for c in candidates:
            desc = (c.get("description") or "").strip()
            if not desc:
                types = ", ".join(c.get("types") or [])
                price = c.get("price_level")
                desc = (
                    f"{c.get('name', '')}. {c.get('category', '')}. "
                    f"rating: {c.get('rating', 'N/A')}. "
                    f"reviews: {c.get('user_ratings_total', 'N/A')}. "
                    f"price_level={price} types={types}. Address={c.get('address','')}"
                )
            texts.append(desc)

        # ✅ Batch embed (1 API call)
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        try:
            pref_emb = client.embeddings.create(
                model="text-embedding-3-small",
                input=user_preferences
            ).data[0].embedding

            cand_embs = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            ).data
        except Exception as e:
            raise RuntimeError(f"Embedding generation failed: {str(e)}")

        # ✅ Compute scores for each candidate
        def cosine(a, b):
            a, b = np.array(a), np.array(b)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        scored = []
        for place, emb_obj in zip(candidates, cand_embs):
            sim = cosine(pref_emb, emb_obj.embedding)

            rating = place.get("rating") or 0.0
            popularity = place.get("user_ratings_total") or 0
            distance = place.get("distance_from_prev") or 0.0

            # ✅ Combine semantic, rating, popularity, distance
            score = (
                0.7 * sim
                + 0.2 * (rating / 5.0)
                + 0.1 * min(popularity / 1000, 1.0)
                - distance_weight * min(distance / 10, 1.0)
            )

            scored.append({**place, "semantic_score": round(score, 3)})

        # ✅ Sort and return top_k
        ranked = sorted(scored, key=lambda x: x["semantic_score"], reverse=True)
        return ranked[:top_k]