# main.py
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
 
from tools.extractor import extractor
from tools.plot_duration import plot_time_duration
from tools.plot_category import plot_service_category_percent
from tools.plot_rootcause import plot_root_cause_stacked_bar
 
 
def main():
    tools = [extractor, plot_time_duration,plot_service_category_percent,plot_root_cause_stacked_bar]
 
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
 
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=(
            "You are an incident analysis assistant with four tools:\n"
            "- extractor: extracts and saves incident records from a report text\n"
            "- plot_time_duration: time duration analysis\n"
            "- plot_service_category_percent: counts of incident for each service\n"
            "- plot_root_cause_stacked_bar: percent of root causes \n"
            "only pick one most suitable tool"
        )
    )
 
    print("=== Incident Analysis Service ===")
    print("Commands: paste a report to extract it, or ask for analysis/chart.")
    print("Type 'exit' to quit.\n")
 
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue
 
        result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        print(f"\nAgent: {result['messages'][-1].content}\n")
 
 
if __name__ == "__main__":
    main()