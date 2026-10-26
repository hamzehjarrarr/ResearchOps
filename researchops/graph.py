"""The research agent: a LangGraph workflow with a follow-up loop.

    planner -> researcher -> extractor -> verifier -> followup
                                 ^                       |
                                 |  new sources found    |
                                 +-----------------------+
                                                         | otherwise
                                                         v
                                                       writer -> END

After each verification pass, the follow-up step checks which report
themes still have too few verified findings. If any do, and the
follow-up budget allows, it searches again for those themes and sends
the new sources back through extraction and verification.
"""

from langgraph.graph import END, StateGraph

from researchops.extractor import extractor
from researchops.planner import planner
from researchops.researcher import followup, researcher, route_after_followup
from researchops.state import State
from researchops.verifier import verifier
from researchops.writer import writer


def build_graph():
    graph = StateGraph(State)

    graph.add_node("planner", planner)
    graph.add_node("researcher", researcher)
    graph.add_node("extractor", extractor)
    graph.add_node("verifier", verifier)
    graph.add_node("followup", followup)
    graph.add_node("writer", writer)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "extractor")
    graph.add_edge("extractor", "verifier")
    graph.add_edge("verifier", "followup")

    graph.add_conditional_edges(
        "followup",
        route_after_followup,
        {"extractor": "extractor", "writer": "writer"},
    )

    graph.add_edge("writer", END)

    return graph.compile()


GRAPH = build_graph()


def run_research(question: str, on_progress=None):
    """
    Run the workflow. If on_progress is given it is called with the
    list of completed agent events after every step, so a caller can
    show live progress.
    """
    initial = {
        "question": question,
        "events": [],
    }

    state = initial

    for state in GRAPH.stream(initial, stream_mode="values"):
        if on_progress:
            try:
                on_progress(state.get("events") or [])
            except Exception:
                pass

    return {
        "question": question,
        "report": state["report"],
        "evidence": state.get("evidence", []),
        "rejected": state.get("rejected", []),
        "sources": state["sources"],
        "verification": state["verification"],
        "plan": state["queries"],
        "followup_queries": state.get("followup_queries", []),
        "themes": state.get("themes", []),
        "followup_rounds": state.get("followup_rounds", 0),
        "review_passed": state["passed"],
        "review_feedback": state["feedback"],
        "events": state["events"],
    }