"""
LangGraph Agent
LangGraph agent — orchestrates all 4 tools using Claude as the brain.
Uses a ReAct-style graph: Reason → Act → Observe → Repeat until done.
"""

from __future__ import annotations
from typing import Annotated, TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agents.tools.weather_tool import get_weather_forecast
from agents.tools.fastf1_tool import get_practice_session_data
from agents.tools.news_tool import get_news_and_penalties
from agents.tools.prediction_tool import run_race_prediction
from agents.prompts.race_analyst import SYSTEM_PROMPT, REPORT_FOOTER
from config import ANTHROPIC_API_KEY, F1_CALENDAR


# Agent State 

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Tools & Model 

TOOLS = [
    get_weather_forecast,
    get_practice_session_data,
    get_news_and_penalties,
    run_race_prediction,
]

llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    api_key=ANTHROPIC_API_KEY,
    temperature=0.2,
    max_tokens=4096,
)

llm_with_tools = llm.bind_tools(TOOLS)


# Graph Nodes 

def call_model(state: AgentState) -> AgentState:
    """Invoke Claude with all tool bindings attached."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """
    Router — if Claude wants to call a tool go to tools node,
    otherwise we are done and go to END.
    """
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# Build Graph 

tool_node = ToolNode(TOOLS)

graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", call_model)
graph_builder.add_node("tools", tool_node)

graph_builder.set_entry_point("agent")
graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END}
)
graph_builder.add_edge("tools", "agent")

agent_graph = graph_builder.compile()


# Public Interface 

def run_pre_race_agent(round_number: int) -> dict:
    """
    Run the full pre-race intelligence pipeline for a given round.

    Args:
        round_number: 2026 F1 calendar round number (1-24).

    Returns:
        dict with final report text, tool outputs, and metadata.
    """
    race_info = F1_CALENDAR.get(round_number)
    if not race_info:
        return {"error": f"Round {round_number} not in 2026 calendar."}

    race_name, location, country, race_date = race_info

    user_prompt = (
        f"Please run a full pre-race analysis for the 2026 {race_name} "
        f"(Round {round_number}) at {location}, {country} on {race_date}. "
        f"Use all 4 tools in order and produce the complete race preview report."
    )

    print(f"\nF1 Primus AI — Starting analysis for Round {round_number}: {race_name}")
    print(f"Location: {location}, Country: {country} | Race Date: {race_date}")
    print("─" * 60)

    final_state = agent_graph.invoke(
        {"messages": [HumanMessage(content=user_prompt)]},
        config={"recursion_limit": 25},
    )

    # Extract report and tool outputs
    report_text  = ""
    tool_outputs = []
    final_ai_message = None

    for msg in final_state["messages"]:
        msg_type = type(msg).__name__
        if msg_type == "AIMessage":
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"Called tool: {tc['name']}")
            else:
                # Track the last non-tool AIMessage only
                final_ai_message = msg
        elif msg_type == "ToolMessage":
            tool_outputs.append({
                "tool":    msg.name,
                "content": msg.content,
            })

    if final_ai_message:
        content = final_ai_message.content
        report_text = content if isinstance(content, str) else str(content)

    report_text += REPORT_FOOTER

    print(f"\nAgent complete — {len(final_state['messages'])} messages exchanged")

    return {
        "round":         round_number,
        "race_name":     race_name,
        "location":      location,
        "race_date":     race_date,
        "report":        report_text,
        "tool_outputs":  tool_outputs,
        "message_count": len(final_state["messages"]),
    }