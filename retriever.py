from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VectorRetriever:
    """
    Simple vector-based retriever using TF-IDF and cosine similarity.

    This component is responsible for:
    - Loading a local text corpus
    - Converting documents into vector representations
    - Ranking documents by semantic similarity to a query

    TF-IDF is used here to avoid platform-specific dependencies
    while still demonstrating real vector-space retrieval.
    """

    def __init__(self, corpus_path: str):
        """
        Initialise the retriever and prepare the corpus.

        Args:
            corpus_path (str): Path to a folder containing .txt files
                               used as the retrieval corpus.
        """
        self.documents = []  # Stores raw document text

        # Load and validate corpus files
        for file in Path(corpus_path).glob("*.txt"):
            # Read file safely, ignoring invalid characters
            text = file.read_text(encoding="utf-8", errors="ignore").strip()

            # Filter out very small or empty documents
            if len(text.split()) > 5:
                self.documents.append(text)

        # Fail fast if the corpus is unusable
        if not self.documents:
            raise ValueError("Corpus is empty or invalid")

        # Configure TF-IDF vectorizer
        # - lowercase: normalize text
        # - stop_words=None: keep all terms for small corpora
        # - token_pattern: allow alphabetic tokens of length >= 2
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=None,
            token_pattern=r"(?u)\b[a-zA-Z]{2,}\b"
        )

        # Convert documents into TF-IDF vectors
        self.doc_vectors = self.vectorizer.fit_transform(self.documents)

    def retrieve(self, query: str, top_k: int = 8):
        """
        Retrieve the top-k most relevant documents for a query.

        Args:
            query (str): User query or task description.
            top_k (int): Number of top-ranked documents to return.

        Returns:
            List[Tuple[str, float]]:
                A list of (document_text, similarity_score),
                sorted by descending relevance.
        """
        # Vectorise the query using the same TF-IDF space
        query_vector = self.vectorizer.transform([query])

        # Compute cosine similarity between query and all documents
        scores = cosine_similarity(query_vector, self.doc_vectors)[0]

        # Rank documents by similarity score (highest first)
        ranked = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Return only the top-k results
        return ranked[:top_k]
