import json
import re
import research_tools
import utils

from aisuite import Client
from datetime import datetime


client = Client()

## ------------------------------------------------------ ##
def find_references(task: str, model: str = "openai:gpt-4o", return_messages: bool = False):
    prompt = f"""
            You are a research function with access to:
            - arxiv_tool: academic papers
            - tavily_tool: general web search (return JSON when asked)
            - wikipedia_tool: encyclopedic summaries

            Task:
            {task}

            Today is {datetime.now().strftime('%Y-%m-%d')}.
            """.strip()

    messages = [{"role" : "user", "content" : prompt}]

    tools = [
            research_tools.arxiv_search_tool,
            research_tools.tavily_search_tool,
            research_tools.wikipedia_search_tool,
            ]

    try:
        response = client.chat.completions.create(
                                                model = model,
                                                messages = messages,
                                                tools = tools,
                                                tool_choice = "auto",
                                                max_turns = 5,
                                                )
        content = response.choices[0].message.content

        return (content, messages) if return_messages else content

    except Exception as e:
        return f"[Model Error: {e}]"

## ------------------------------------------------------ ##
research_task = "Find 2 recent papers about recent developments in black hole science"
research_result = find_references(research_task)

utils.print_html(research_result, title = "Research Function Output")

## ------------------------------------------------------ ##
TOP_DOMAINS = {
            "wikipedia.org", "nature.com", "science.org", "sciencemag.org", "cell.com",
            "mit.edu", "stanford.edu", "harvard.edu", "nasa.gov", "noaa.gov", "europa.eu",

            "arxiv.org", "acm.org", "ieee.org", "neurips.cc", "icml.cc", "openreview.net",

            "elifesciences.org", "pnas.org", "jmlr.org", "springer.com", "sciencedirect.com",

            "pbs.org", "nova.edu", "nvcc.edu", "cccco.edu",

            "codecademy.com", "datacamp.com"
            }

def evaluate_tavily_results(TOP_DOMAINS, raw: str, min_ratio = 0.4):
    url_pattern = re.compile(r'https?://[^\s\]\)>\}]+', flags = re.IGNORECASE)
    urls = url_pattern.findall(raw)

    if not urls:
        return False, """### Evaluation — Tavily Preferred Domains
                        No URLs detected in the provided text.
                        Please include links in your research results.
                        """

    total = len(urls)
    preferred_count = 0
    details = []

    for url in urls:
        domain = url.split("/")[2]
        preferred = any(td in domain for td in TOP_DOMAINS)

        if preferred:
            preferred_count += 1

        details.append(f"- {url} → {'✅ PREFERRED' if preferred else '❌ NOT PREFERRED'}")

    ratio = preferred_count / total if total > 0 else 0.0
    flag = ratio >= min_ratio

    report = f"""
            ### Evaluation — Tavily Preferred Domains
            - Total results: {total}
            - Preferred results: {preferred_count}
            - Ratio: {ratio:.2%}
            - Threshold: {min_ratio:.0%}
            - Status: {"✅ PASS" if flag else "❌ FAIL"}

            **Details:**
            {chr(10).join(details)}
            """

    return flag, report

## ------------------------------------------------------ ##
utils.print_html(json.dumps(list(TOP_DOMAINS)[ : 4], indent = 2), title = "Sample Trusted Domains")

utils.print_html("<h3>Research Results</h3>" + research_result, title = "Research Results")

flag, report = evaluate_tavily_results(TOP_DOMAINS, research_result)
utils.print_html("<pre>" + report + "</pre>", title="<h3>Evaluation Summary</h3>")

## ------------------------------------------------------ ##
topic = ("Latest breakthroughs in Tokamak plasma confinement techniques for practical nuclear "
        "fusion energy.")
min_ratio = 0.34
run_reflection = True

TOP_DOMAINS = {
                "wikipedia.org", "nature.com", "science.org", "arxiv.org",
                "nasa.gov", "mit.edu", "stanford.edu", "harvard.edu"
                }

utils.print_html(
                json.dumps(sorted(list(TOP_DOMAINS)), indent = 2),
                title = "<h3>Sample Preferred Domains</h3>"
                )

research_task = f"Find 2–3 key papers and reliable overviews about {topic}."
research_output = find_references(research_task)
utils.print_html(research_output, title = f"<h3>Research Results on {topic}</h3>")

flag, eval_md = evaluate_tavily_results(TOP_DOMAINS, research_output, min_ratio = min_ratio)
utils.print_html("<pre>" + eval_md + "</pre>", title = "<h3>Evaluation Summary</h3>")

## ------------------------------------------------------ ##
topic = "recent advancements in fusion energy and the development of plasma-based tokamak reactors"
min_ratio = 0.15
run_reflection = True

TOP_DOMAINS = {
                "wikipedia.org", "nature.com", "science.org", "arxiv.org",
                "nasa.gov", "mit.edu", "stanford.edu", "harvard.edu"
                }

utils.print_html(
                json.dumps(sorted(list(TOP_DOMAINS)), indent = 2),
                title = "<h3>Sample Preferred Domains</h3>"
                )

research_task = f"Find 2-3 key papers and reliable overviews about {topic}."
research_output = find_references(research_task)
utils.print_html(research_output, title = f"<h3>Research Results on {topic}</h3>")

flag, eval_md = evaluate_tavily_results(TOP_DOMAINS, research_output, min_ratio = min_ratio)
utils.print_html("<pre>" + eval_md + "</pre>", title = "<h3>Evaluation Summary</h3>")
