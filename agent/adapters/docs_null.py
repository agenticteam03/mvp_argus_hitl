"""
what: a DocsPort adapter that retrieves nothing.
why:  the graph calls docs.search() from Stage 0 onward, so the seam is exercised
      from the first run rather than being retrofitted in Stage 4. Wiring a port
      in early and leaving it empty costs four lines; adding a port to a working
      graph later costs an argument about where it belongs.
how:  search() returns []. Every consumer must already cope with "no runbook
      chunks found", because a real retriever returns [] too -- on an empty
      corpus, or on a query that matches nothing.

      Stage 4 replaces this with docs_tfidf.py, and the swap is one line in
      run.py. Nothing else in the project knows this class existed.
"""


class NullDocs:
    """Retrieves nothing. Satisfies DocsPort."""

    def search(self, query: str, k: int = 3) -> list[str]:
        return []
