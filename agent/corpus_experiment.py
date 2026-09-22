"""
what: the same retriever, the same question, three different corpora.
why:  mvp_plan.md asks the team to prove one claim to themselves: retrieval
      quality is decided by WHICH documents you index, not by how clever the
      retriever is. Reading that is not the same as watching it.
how:  ask each corpus the real incident summary and print the top chunks.

      No model call, so running this costs nothing. It lives in a module rather
      than a one-line command because the one-liner version shipped with a
      syntax error inside a tuple, and nobody could read it well enough to see.
"""

from agent.adapters.docs_tfidf import TfidfDocs
from agent.adapters.metrics_fake import FakeMetrics
from agent.detectors import detect

# (label, path). A is what the agent actually uses.
CORPORA = [
    ("A  RB-002 alone (what the agent reads)", "runbooks/RB-002-container-restart-loop.md"),
    ("B  all three runbooks (related AND relevant)", "../02_PlanningFirstIdea/runbooks"),
    (
        "C  a generative-AI course (plausible, irrelevant)",
        "../../../practice_before_agents/generative-ai-for-beginners",
    ),
]


def main(k: int = 3) -> None:
    # The real query: exactly what recall_node sends, not a tidied-up version.
    query = detect(FakeMetrics().container_stats("lab-victim")).summary
    print("QUERY (the incident summary, as recall_node sends it):")
    print(f"  {query}\n")

    for label, path in CORPORA:
        try:
            docs = TfidfDocs(path)
        except (FileNotFoundError, ValueError) as unreadable:
            # A missing corpus is worth saying out loud rather than crashing the
            # whole comparison: the other two still have something to teach.
            print(f"  {label}\n     unavailable: {unreadable}\n")
            continue
        print(f"  {label}  ->  {len(docs.chunks)} chunks indexed")
        for chunk in docs.search(query, k=k):
            print(f"     {chunk.splitlines()[0][:76]}")
        print()


if __name__ == "__main__":
    main()
