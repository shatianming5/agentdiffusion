"""News conditioning module for Rich Agent system.

Loads financial news from 2024_News_Security.xlsx, computes per-stock
text embeddings (TF-IDF + PCA, 32-dim) and keyword-based sentiment
scores, and provides conditioning tensors for the Video DiT pipeline.

News conditioning is STATIC per day: every time window in the same
trading day receives the same news embedding. It is concatenated
to the 8-dim market_cond to form a 40-dim conditioning vector.

Stocks absent from the news dataset receive zero vectors, which the
model learns to interpret as "no news = neutral".
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

D_NEWS = 32  # output embedding dimension

POSITIVE_KEYWORDS = ["涨", "利好", "增长", "突破", "新高", "盈利", "分红"]
NEGATIVE_KEYWORDS = ["跌", "利空", "下跌", "爆雷", "退市", "亏损", "减持", "停产"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_stock_code(raw_code: str) -> str:
    """Extract 6-digit numeric code from various formats.

    Handles: '000001', '000001.SZ', 'SZ000001', '1', etc.
    Returns e.g. '000001'.
    """
    digits = "".join(c for c in str(raw_code) if c.isdigit())
    if len(digits) == 0:
        return ""
    return digits.zfill(6)


def _tokenise(text: str, use_jieba: bool = False) -> list[str]:
    """Tokenise Chinese text into words (jieba) or characters (fallback)."""
    if use_jieba:
        try:
            import jieba
            return list(jieba.cut(text))
        except ImportError:
            pass
    # Character-level fallback (skip ASCII punctuation / spaces)
    return [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]


def _build_tfidf_matrix(
    documents: list[list[str]],
    max_features: int = 2000,
) -> tuple[np.ndarray, list[str]]:
    """Compute a simple TF-IDF matrix from pre-tokenised documents.

    Returns:
        tfidf: [n_docs, n_features] float64
        vocab: list of feature tokens
    """
    # Build vocabulary from document frequency
    from collections import Counter

    doc_freq: Counter = Counter()
    for doc in documents:
        doc_freq.update(set(doc))

    # Keep the top max_features tokens by document frequency
    vocab = [tok for tok, _ in doc_freq.most_common(max_features)]
    tok2idx = {tok: i for i, tok in enumerate(vocab)}
    n_docs = len(documents)
    n_feat = len(vocab)

    if n_feat == 0:
        return np.zeros((n_docs, 1), dtype=np.float64), []

    # Term frequency (raw counts per document)
    tf = np.zeros((n_docs, n_feat), dtype=np.float64)
    for d, doc in enumerate(documents):
        for tok in doc:
            idx = tok2idx.get(tok)
            if idx is not None:
                tf[d, idx] += 1
        # Sub-linear TF: 1 + log(tf) if tf > 0
        mask = tf[d] > 0
        tf[d, mask] = 1.0 + np.log(tf[d, mask])

    # IDF
    df = np.array([doc_freq.get(tok, 0) for tok in vocab], dtype=np.float64)
    idf = np.log((1 + n_docs) / (1 + df)) + 1  # sklearn-style smooth IDF

    tfidf = tf * idf[np.newaxis, :]

    # L2 normalise each document vector
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    tfidf = tfidf / norms

    return tfidf, vocab


def _pca_reduce(X: np.ndarray, n_components: int) -> np.ndarray:
    """Reduce dimensionality via truncated SVD (PCA without centering bias).

    Returns: [n_samples, n_components] float32
    """
    if X.shape[1] <= n_components:
        pad = np.zeros((X.shape[0], n_components - X.shape[1]), dtype=np.float32)
        return np.hstack([X.astype(np.float32), pad])

    # Centre
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    # Truncated SVD via np (fine for <2000 features)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (U[:, :n_components] * S[:n_components]).astype(np.float32)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class NewsConditioner:
    """Per-stock news embedding and sentiment for a single trading day.

    Parameters
    ----------
    news_excel_path : str or Path
        Path to 2024_News_Security.xlsx.
    target_date : str
        Target date in 'YYYY-MM-DD' or 'YYYYMMDD' format.
    d_news : int
        Output embedding dimension (default 32).
    max_tfidf_features : int
        Maximum vocabulary size for TF-IDF (default 2000).
    """

    def __init__(
        self,
        news_excel_path: str | Path,
        target_date: str = "2024-06-19",
        d_news: int = D_NEWS,
        max_tfidf_features: int = 2000,
    ):
        self.d_news = d_news
        self.news_excel_path = Path(news_excel_path)
        self.target_date_str = target_date.replace("-", "")  # -> '20240619'

        # Detect jieba availability once
        self._use_jieba = False
        try:
            import jieba  # noqa: F401
            self._use_jieba = True
            logger.info("NewsConditioner: using jieba for Chinese tokenisation")
        except ImportError:
            logger.info(
                "NewsConditioner: jieba not available, using character-level tokenisation"
            )

        # Load and filter news
        self._load_news()

        # Build embeddings
        self._build_embeddings(max_tfidf_features)

        # Compute sentiment
        self._compute_sentiment()

        logger.info(
            "NewsConditioner ready: %d stocks with news, d_news=%d, date=%s",
            len(self.stock_news), self.d_news, self.target_date_str,
        )

    # ------------------------------------------------------------------
    # Internal: data loading
    # ------------------------------------------------------------------

    def _load_news(self) -> None:
        """Load xlsx, filter to target date, group titles by stock code."""
        import pandas as pd

        if not self.news_excel_path.exists():
            logger.warning(
                "News file not found: %s — all stocks will get zero embeddings",
                self.news_excel_path,
            )
            self.stock_news: dict[str, list[str]] = {}
            self.stock_full_dates: dict[str, list[str]] = {}
            return

        logger.info("Loading news from %s ...", self.news_excel_path)
        df = pd.read_excel(
            self.news_excel_path,
            skiprows=2,
            engine="openpyxl",
        )

        # Expected columns after skiprows=2:
        # NewsID, DeclareDate, Title, Symbol, ShortName,
        # SecurityTypeID, SecurityType, FullDeclareDate
        # Rename by position for robustness
        expected_cols = [
            "NewsID", "DeclareDate", "Title", "Symbol", "ShortName",
            "SecurityTypeID", "SecurityType", "FullDeclareDate",
        ]
        if len(df.columns) >= len(expected_cols):
            df.columns = expected_cols[:len(df.columns)]
        else:
            # Try to use whatever columns are present
            logger.warning(
                "News file has %d columns, expected %d",
                len(df.columns), len(expected_cols),
            )

        # Normalise DeclareDate to string YYYYMMDD
        df["_date_str"] = df["DeclareDate"].astype(str).str.replace("-", "").str[:8]

        # Filter to target date
        mask = df["_date_str"] == self.target_date_str
        day_df = df[mask].copy()
        logger.info(
            "News for %s: %d items covering %d symbols",
            self.target_date_str,
            len(day_df),
            day_df["Symbol"].nunique() if "Symbol" in day_df.columns else 0,
        )

        # Group by normalised stock code
        self.stock_news = {}       # code -> [title, ...]
        self.stock_full_dates = {} # code -> [FullDeclareDate, ...]

        for _, row in day_df.iterrows():
            code = _normalise_stock_code(str(row.get("Symbol", "")))
            title = str(row.get("Title", ""))
            full_dt = str(row.get("FullDeclareDate", ""))

            if not code or not title:
                continue

            self.stock_news.setdefault(code, []).append(title)
            self.stock_full_dates.setdefault(code, []).append(full_dt)

    # ------------------------------------------------------------------
    # Internal: text embedding via TF-IDF + PCA
    # ------------------------------------------------------------------

    def _build_embeddings(self, max_features: int) -> None:
        """Build d_news-dim embeddings from concatenated news titles per stock."""
        codes = sorted(self.stock_news.keys())
        if len(codes) == 0:
            self._embedding_map: dict[str, np.ndarray] = {}
            return

        # One document per stock = all titles concatenated
        documents = []
        for code in codes:
            all_text = " ".join(self.stock_news[code])
            tokens = _tokenise(all_text, use_jieba=self._use_jieba)
            documents.append(tokens)

        tfidf, vocab = _build_tfidf_matrix(documents, max_features=max_features)
        logger.info(
            "TF-IDF matrix: %d stocks x %d features", tfidf.shape[0], tfidf.shape[1],
        )

        # PCA to d_news
        reduced = _pca_reduce(tfidf, self.d_news)  # [n_stocks, d_news]

        self._embedding_map = {
            code: reduced[i] for i, code in enumerate(codes)
        }

    # ------------------------------------------------------------------
    # Internal: keyword-based sentiment
    # ------------------------------------------------------------------

    def _compute_sentiment(self) -> None:
        """Compute per-stock sentiment score from keyword matching."""
        self.sentiment_map: dict[str, float] = {}

        for code, titles in self.stock_news.items():
            pos_count = 0
            neg_count = 0
            for title in titles:
                for kw in POSITIVE_KEYWORDS:
                    if kw in title:
                        pos_count += 1
                for kw in NEGATIVE_KEYWORDS:
                    if kw in title:
                        neg_count += 1
            total = len(titles)
            score = (pos_count - neg_count) / max(total, 1)
            self.sentiment_map[code] = score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_stock_embeddings(self, stock_codes: list[str]) -> np.ndarray:
        """Return [n_stocks, d_news] news embeddings for given stocks.

        Stocks without news on the target date receive zero vectors.
        """
        n = len(stock_codes)
        result = np.zeros((n, self.d_news), dtype=np.float32)

        for i, raw_code in enumerate(stock_codes):
            code = _normalise_stock_code(raw_code)
            emb = self._embedding_map.get(code)
            if emb is not None:
                result[i] = emb

        return result

    def get_stock_sentiments(self, stock_codes: list[str]) -> np.ndarray:
        """Return [n_stocks] sentiment scores for given stocks.

        Stocks without news on the target date get 0.0 (neutral).
        """
        result = np.zeros(len(stock_codes), dtype=np.float32)
        for i, raw_code in enumerate(stock_codes):
            code = _normalise_stock_code(raw_code)
            result[i] = self.sentiment_map.get(code, 0.0)
        return result

    def get_grid_conditioning(
        self,
        stock_codes: list[str],
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """Return [grid_h, grid_w, d_news] news conditioning for the grid.

        stock_codes should have exactly grid_h * grid_w entries, in row-major
        order matching the grid layout.
        """
        n_cells = grid_h * grid_w
        # Pad or truncate codes to match grid
        codes = list(stock_codes)
        if len(codes) < n_cells:
            codes += [""] * (n_cells - len(codes))
        codes = codes[:n_cells]

        embeddings = self.get_stock_embeddings(codes)  # [n_cells, d_news]
        return torch.from_numpy(
            embeddings.reshape(grid_h, grid_w, self.d_news)
        ).float()

    def get_batch_conditioning(
        self,
        stock_codes: list[str],
        grid_h: int,
        grid_w: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Return [B, d_news] news conditioning (mean across grid) for batches.

        Since news is static per day, the same vector is replicated B times.
        This is intended to be concatenated with the 8-dim market_cond.
        """
        grid_cond = self.get_grid_conditioning(stock_codes, grid_h, grid_w)
        # Average across grid to get a single d_news vector
        mean_cond = grid_cond.mean(dim=(0, 1))  # [d_news]
        return mean_cond.unsqueeze(0).expand(batch_size, -1)  # [B, d_news]

    def get_per_stock_batch_conditioning(
        self,
        stock_codes: list[str],
        batch_size: int,
    ) -> torch.Tensor:
        """Return [B, n_stocks, d_news] per-stock news conditioning.

        Replicated across batch dimension since news is static.
        """
        embeddings = self.get_stock_embeddings(stock_codes)  # [n_stocks, d_news]
        t = torch.from_numpy(embeddings).float()  # [n_stocks, d_news]
        return t.unsqueeze(0).expand(batch_size, -1, -1)  # [B, n_stocks, d_news]

    def flip_sentiment(self) -> None:
        """Flip sentiment for counterfactual experiments.

        Inverts all embeddings (multiply by -1) and negates sentiment scores.
        This allows testing whether the model reacts to news direction.
        """
        for code in self._embedding_map:
            self._embedding_map[code] = -self._embedding_map[code]
        for code in self.sentiment_map:
            self.sentiment_map[code] = -self.sentiment_map[code]
        logger.info("NewsConditioner: sentiment FLIPPED for counterfactual")

    def summary(self, stock_codes: list[str] | None = None) -> dict:
        """Return a summary dict of news statistics."""
        codes_with_news = len(self.stock_news)
        total_articles = sum(len(v) for v in self.stock_news.values())
        sentiments = list(self.sentiment_map.values())
        info = {
            "date": self.target_date_str,
            "stocks_with_news": codes_with_news,
            "total_articles": total_articles,
            "d_news": self.d_news,
            "mean_sentiment": float(np.mean(sentiments)) if sentiments else 0.0,
            "std_sentiment": float(np.std(sentiments)) if sentiments else 0.0,
        }
        if stock_codes is not None:
            matched = 0
            for c in stock_codes:
                code = _normalise_stock_code(c)
                if code in self.stock_news:
                    matched += 1
            info["grid_stocks_with_news"] = matched
            info["grid_stocks_total"] = len(stock_codes)
        return info
