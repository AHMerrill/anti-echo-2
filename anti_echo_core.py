"""
Anti-Echo Chamber Core Library
Consolidated functions for news article processing, embedding, and classification.
"""

import os
import json
import yaml
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    import nltk
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    from nltk.tokenize import sent_tokenize
    import re
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    pass

class AntiEchoCore:
    """Core functionality for the Anti-Echo Chamber system."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the core system with configuration."""
        self.config = self.load_config(config_path)
        self.device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        
        # Initialize models lazily
        self._topic_embedder = None
        self._stance_embedder = None
        self._flan_classifier = None
        self._bart_summarizer = None
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Load topic and stance configurations
        self.topics = self.load_json("config/topics.json")
        self.political_leanings = self.load_json("config/political_leanings.json")
        self.implied_stances = self.load_json("config/implied_stances.json")
        
        # Create collections
        self.topic_coll = self.chroma_client.get_or_create_collection(
            name="news_topic",
            metadata={"hnsw:space": "cosine"}
        )
        self.stance_coll = self.chroma_client.get_or_create_collection(
            name="news_stance", 
            metadata={"hnsw:space": "cosine"}
        )
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_json(self, file_path: str) -> Dict:
        """Load JSON configuration file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @property
    def topic_embedder(self):
        """Lazy load topic embedding model."""
        if self._topic_embedder is None:
            model_name = self.config["embeddings"]["topic_model"]
            self._topic_embedder = SentenceTransformer(model_name, device=self.device)
        return self._topic_embedder
    
    @property
    def stance_embedder(self):
        """Lazy load stance embedding model."""
        if self._stance_embedder is None:
            model_name = self.config["embeddings"]["stance_model"]
            self._stance_embedder = SentenceTransformer(model_name, device=self.device)
        return self._stance_embedder
    
    @property
    def flan_classifier(self):
        """Lazy load FLAN-T5 classifier."""
        if self._flan_classifier is None:
            model_name = self.config["stance_processing"]["classifier"]["model"]
            self._flan_classifier = pipeline(
                "text2text-generation",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
                max_length=512
            )
        return self._flan_classifier
    
    @property
    def bart_summarizer(self):
        """Lazy load BART summarizer."""
        if self._bart_summarizer is None:
            model_name = self.config["stance_processing"]["summarizer"]["model"]
            self._bart_summarizer = pipeline(
                "summarization",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
                max_length=512
            )
        return self._bart_summarizer
    
    def sent_split(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        try:
            return sent_tokenize(text)
        except:
            # Fallback to simple sentence splitting
            return re.split(r'[.!?]+', text)
    
    def chunk_by_tokens(self, sentences: List[str], max_tokens: int = 512) -> List[str]:
        """Chunk sentences into token-limited chunks."""
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            # Rough token estimation (4 chars per token)
            sentence_tokens = len(sentence) // 4
            
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def encode_text(self, text: str, embedder) -> np.ndarray:
        """Encode text using the specified embedder."""
        return embedder.encode(text, convert_to_numpy=True)
    
    def topic_vecs(self, text: str) -> np.ndarray:
        """Generate topic vectors for text using clustering approach."""
        sentences = self.sent_split(text)
        chunks = self.chunk_by_tokens(sentences, self.config["topics"]["max_tokens_per_chunk"])
        
        if not chunks:
            return np.array([])
        
        # Encode chunks
        chunk_embeddings = self.encode_text(chunks, self.topic_embedder)
        
        # Cluster chunks
        if len(chunk_embeddings) > 1:
            clustering = AgglomerativeClustering(
                n_clusters=min(self.config["topics"]["max_topics_per_article"], len(chunk_embeddings)),
                metric='cosine',
                linkage='average'
            )
            cluster_labels = clustering.fit_predict(chunk_embeddings)
            
            # Get cluster centroids
            centroids = []
            for cluster_id in range(max(cluster_labels) + 1):
                cluster_embeddings = chunk_embeddings[cluster_labels == cluster_id]
                centroid = np.mean(cluster_embeddings, axis=0)
                centroids.append(centroid)
            
            return np.array(centroids)
        else:
            return chunk_embeddings
    
    def match_topics(self, vec: np.ndarray) -> List[str]:
        """Match topic vectors to predefined topic anchors."""
        if len(vec) == 0:
            return []
        
        topic_anchors = self.topics["anchors"]
        topic_names = list(topic_anchors.keys())
        anchor_embeddings = np.array(list(topic_anchors.values()))
        
        # Calculate similarities
        similarities = cosine_similarity(vec.reshape(1, -1), anchor_embeddings)[0]
        
        # Find matches above threshold
        threshold = self.config["topics"]["similarity_threshold"]
        matches = []
        for i, sim in enumerate(similarities):
            if sim > threshold:
                matches.append(topic_names[i])
        
        return matches
    
    def flan_classify(self, text: str) -> Tuple[str, str]:
        """Classify political leaning and implied stance using FLAN-T5."""
        prompt = f"""
        Analyze this news article text and classify it:

        Text: {text[:1000]}

        Political Leaning Options: {list(self.political_leanings.keys())}
        Implied Stance Options: {list(self.implied_stances.keys())}

        Respond in JSON format:
        {{"political_leaning": "option", "implied_stance": "option"}}
        """
        
        try:
            result = self.flan_classifier(prompt, max_length=100, do_sample=False)
            response = result[0]["generated_text"]
            
            # Parse JSON response
            try:
                parsed = json.loads(response)
                political_leaning = parsed.get("political_leaning", "apolitical_or_unknown")
                implied_stance = parsed.get("implied_stance", "unknown")
            except:
                political_leaning = "apolitical_or_unknown"
                implied_stance = "unknown"
                
        except Exception as e:
            print(f"Classification error: {e}")
            political_leaning = "apolitical_or_unknown"
            implied_stance = "unknown"
        
        return political_leaning, implied_stance
    
    def bart_one_sentence(self, text: str) -> str:
        """Generate one-sentence summary using BART."""
        try:
            # Truncate text to avoid token limits
            max_length = 1000
            if len(text) > max_length:
                text = text[:max_length]
            
            result = self.bart_summarizer(text, max_length=50, min_length=10, do_sample=False)
            return result[0]["summary_text"]
        except Exception as e:
            print(f"Summarization error: {e}")
            return "Summary generation failed"
    
    def process_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single article through the full pipeline."""
        text = article.get("text", "")
        if not text:
            return None
        
        # Topic analysis
        topic_vecs = self.topic_vecs(text)
        topic_matches = []
        if len(topic_vecs) > 0:
            for vec in topic_vecs:
                matches = self.match_topics(vec)
                topic_matches.extend(matches)
        
        # Stance classification
        political_leaning, implied_stance = self.flan_classify(text)
        summary = self.bart_one_sentence(text)
        
        # Create stance embedding
        stance_text = f"Political: {political_leaning}. Stance: {implied_stance}. Summary: {summary}"
        stance_embedding = self.encode_text(stance_text, self.stance_embedder)
        
        # Create article ID
        article_id = hashlib.md5(text.encode()).hexdigest()
        
        return {
            "id": article_id,
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "source": article.get("source", ""),
            "published": article.get("published", ""),
            "topics": list(set(topic_matches)),
            "political_leaning": political_leaning,
            "implied_stance": implied_stance,
            "summary": summary,
            "topic_vectors": topic_vecs.tolist() if len(topic_vecs) > 0 else [],
            "stance_embedding": stance_embedding.tolist(),
            "text_length": len(text)
        }
    
    def upsert_to_chroma(self, processed_article: Dict[str, Any]):
        """Upsert processed article to ChromaDB collections."""
        article_id = processed_article["id"]
        
        # Upsert to topic collection
        if processed_article["topic_vectors"]:
            self.topic_coll.upsert(
                ids=[f"{article_id}_topic"],
                embeddings=[processed_article["topic_vectors"][0].tolist()],
                metadatas=[{
                    "article_id": article_id,
                    "title": processed_article["title"],
                    "source": processed_article["source"],
                    "topics": processed_article["topics"],
                    "political_leaning": processed_article["political_leaning"],
                    "implied_stance": processed_article["implied_stance"]
                }]
            )
        
        # Upsert to stance collection
        self.stance_coll.upsert(
            ids=[f"{article_id}_stance"],
            embeddings=[processed_article["stance_embedding"]],
            metadatas=[{
                "article_id": article_id,
                "title": processed_article["title"],
                "source": processed_article["source"],
                "political_leaning": processed_article["political_leaning"],
                "implied_stance": processed_article["implied_stance"],
                "summary": processed_article["summary"]
            }]
        )
    
    def export_metadata_only(self, processed_article: Dict[str, Any]) -> Dict[str, Any]:
        """Export only metadata and embeddings (no full text) for HF/Git storage."""
        return {
            "id": processed_article["id"],
            "title": processed_article["title"],
            "url": processed_article["url"],
            "source": processed_article["source"],
            "published": processed_article["published"],
            "topics": processed_article["topics"],
            "political_leaning": processed_article["political_leaning"],
            "implied_stance": processed_article["implied_stance"],
            "summary": processed_article["summary"],
            "topic_vectors": processed_article["topic_vectors"],
            "stance_embedding": processed_article["stance_embedding"],
            "text_length": processed_article["text_length"]
            # Note: No 'text' field - only metadata and embeddings
        }
    
    def query_similar_articles(self, query_text: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Query for similar articles based on topic similarity."""
        # Generate topic vectors for query
        query_topic_vecs = self.topic_vecs(query_text)
        if len(query_topic_vecs) == 0:
            return []
        
        # Query topic collection
        results = self.topic_coll.query(
            query_embeddings=[query_topic_vecs[0].tolist()],
            n_results=n_results,
            include=["metadatas", "distances"]
        )
        
        return results["metadatas"][0] if results["metadatas"] else []
    
    def query_opposing_stance(self, query_text: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Query for articles with opposing political stance."""
        # Generate stance embedding for query
        query_stance_embedding = self.encode_text(query_text, self.stance_embedder)
        
        # Query stance collection
        results = self.stance_coll.query(
            query_embeddings=[query_stance_embedding.tolist()],
            n_results=n_results * 2,  # Get more results to filter
            include=["metadatas", "distances"]
        )
        
        # Filter for different political leanings
        opposing_articles = []
        if results["metadatas"]:
            for metadata in results["metadatas"][0]:
                # Simple heuristic: different political leaning = opposing
                if metadata.get("political_leaning") != "apolitical_or_unknown":
                    opposing_articles.append(metadata)
                if len(opposing_articles) >= n_results:
                    break
        
        return opposing_articles
