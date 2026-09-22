"""
what: tests for the TF-IDF runbook retriever.
why:  two things need pinning. That the RIGHT question finds the right section
      of RB-002 -- above all the DO NOT list, which contradicts what the model
      has recommended in every real run so far. And that a nonsense question
      still gets confident chunks back, because that silent failure is the whole
      point of the Stage 4 corpus experiment.
how:  the real runbook is the corpus, since it is checked in and never changes
      under the tests. One test builds a tiny corpus in tmp_path to prove that a
      directory of several files works, which is what the wrong-corpus
      experiment needs.
"""

from agent.adapters.docs_tfidf import TfidfDocs
from agent.ports import DocsPort

RUNBOOK = "runbooks/RB-002-container-restart-loop.md"


def test_the_adapter_satisfies_the_port():
    # Given:    a retriever over the real runbook
    # Expected: it passes isinstance against DocsPort
    # Why:      NullDocs and this must be interchangeable in run.py
    assert isinstance(TfidfDocs(RUNBOOK), DocsPort)


def test_it_chunks_on_h2_headings():
    # Given:    RB-002, which has several ## sections
    # Expected: more than four chunks, each starting with its file name and ##
    # Why:      a chunk without its heading is much harder for a model to cite
    docs = TfidfDocs(RUNBOOK)
    assert len(docs.chunks) > 4
    assert all(chunk.startswith("[RB-002-container-restart-loop.md] ## ") for chunk in docs.chunks)


def test_asking_about_the_symptom_finds_the_exit_code_table():
    # Given:    a query about an OOM-killed container in a restart loop
    # Expected: a chunk mentioning exit code 137
    # Why:      the detector's evidence and the runbook's table must meet
    found = TfidfDocs(RUNBOOK).search("container OOM killed exit 137 restart loop", k=3)
    assert any("137" in chunk for chunk in found)


def test_asking_about_raising_the_limit_finds_the_do_not_list():
    # Given:    the exact remediation every real run has proposed
    # Expected: the DO NOT section, which forbids it above ~1 GiB
    # Why:      THE test of this stage: the runbook must be able to contradict the model
    found = TfidfDocs(RUNBOOK).search("should I raise the memory limit to fix this?", k=3)
    assert any("Never raise a memory limit" in chunk for chunk in found)


def test_k_limits_how_much_reaches_the_prompt():
    # Given:    the same query at k=1 and k=3
    # Expected: one chunk, then three
    # Why:      every chunk is prompt tokens, and tokens are the budget
    docs = TfidfDocs(RUNBOOK)
    assert len(docs.search("memory", k=1)) == 1
    assert len(docs.search("memory", k=3)) == 3


def test_a_nonsense_query_still_returns_confident_chunks():
    # Given:    a query with nothing to do with containers
    # Expected: k chunks anyway, no error, no warning
    # Why:      a ranking has no "I don't know" -- this is the silent failure
    #           the wrong-corpus experiment is about to make visible
    found = TfidfDocs(RUNBOOK).search("banana pancake recipe for breakfast", k=2)
    assert len(found) == 2


def test_a_directory_of_documents_is_one_corpus(tmp_path):
    # Given:    two markdown files in a folder
    # Expected: chunks from both, each tagged with its own file name
    # Why:      the Stage 4 experiment points this same class at another folder
    (tmp_path / "one.md").write_text("# One\n\n## Alpha\nDisk pressure and inodes.\n", encoding="utf-8")
    (tmp_path / "two.md").write_text("# Two\n\n## Beta\nPostgres connection limits.\n", encoding="utf-8")
    docs = TfidfDocs(str(tmp_path))
    names = {chunk.split("]")[0] + "]" for chunk in docs.chunks}
    assert names == {"[one.md]", "[two.md]"}
    assert "Postgres" in docs.search("postgres connections", k=1)[0]
