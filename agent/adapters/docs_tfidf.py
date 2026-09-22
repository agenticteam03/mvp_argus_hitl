"""
what: a DocsPort adapter that retrieves runbook sections with TF-IDF.
why:  the agent's three real runs all recommended raising the memory limit --
      the one thing RB-002's DO NOT list forbids. The model is not wrong-headed;
      it has simply never read the runbook. This is the file that hands it the
      relevant paragraphs.

      TF-IDF and not embeddings, deliberately. It is ten lines, costs nothing,
      needs no API and no model download -- and retrieval quality here is
      dominated by WHICH corpus you point at, not by how clever the retriever
      is. Stage 4's experiment proves that by pointing this same class at the
      wrong corpus and watching it answer confidently anyway.
how:  split each document on its `##` headings, score chunks against the query
      by cosine similarity of TF-IDF vectors, return the best k.

      Every chunk is prefixed with its file name, so a model quoting a chunk can
      cite RB-002 by name -- which is exactly what mvp_plan.md asks Stage 4 to
      produce.
"""

import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_PATH = "runbooks/RB-002-container-restart-loop.md"

# A line that starts with exactly two hashes. Splitting on ## and not ### keeps
# "Likely causes" whole instead of scattering its four numbered causes.
HEADING = re.compile(r"^## ", re.MULTILINE)


class TfidfDocs:
    """Retrieval over local markdown. Satisfies DocsPort."""

    def __init__(self, path: str = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.chunks = self._load_chunks()
        if not self.chunks:
            raise ValueError(f"No readable markdown under {path}")
        # stop_words="english" drops "the", "and", "is" -- words that appear in
        # every chunk carry no signal about which chunk you want.
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(self.chunks)

    def search(self, query: str, k: int = 3) -> list[str]:
        """The k chunks closest to the query.

        Note what this NEVER does: return nothing. TF-IDF always ranks, so even
        a query with no real match gets the k least-bad chunks, at whatever tiny
        similarity. There is no "I don't know" in a ranking, and that is the
        silent failure mode Stage 4 exists to show you.
        """
        scores = (self._vectorizer.transform([query]) @ self._matrix.T).toarray()[0]
        # argsort is ascending, so the last k are the best; reversed puts the
        # strongest match first, which is what a prompt should lead with.
        best = scores.argsort()[-k:][::-1]
        return [self.chunks[index] for index in best]

    def _load_chunks(self) -> list[str]:
        """Every `##` section of every markdown file in the corpus."""
        files = sorted(self.path.rglob("*.md")) if self.path.is_dir() else [self.path]
        chunks: list[str] = []
        for document in files:
            text = document.read_text(encoding="utf-8", errors="replace")
            # re.split drops the "## " marker, so it is put back below -- a
            # chunk that has lost its heading is much harder for a model to cite.
            sections = HEADING.split(text)
            for section in sections[1:]:  # [0] is the frontmatter and the title
                body = section.strip()
                if body:
                    chunks.append(f"[{document.name}] ## {body}")
        return chunks
