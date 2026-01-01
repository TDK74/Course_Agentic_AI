import matplotlib.pyplot as plt
import pandas as pd
import os, re

from typing import Dict, Any
from utils import *


## ------------------------------------------------------ ##
df = load_and_prepare_data('coffee_sales.csv')

print_html(df.sample(n = 5), title = "Random Sample of Coffee Sales Data")

## ------------------------------------------------------ ##
def get_response(model: str, prompt: str) -> str:
    if "claude" in model.lower() or "anthropic" in model.lower():
        message = anthropic_client.messages.create(model = model,
                                                   max_tokens = 1000,
                                                   messages = [{"role" : "user",
                                                                "content" : [{"type" : "text",
                                                                              "text" : prompt}]
                                                                }]
                                                    )

        return message.content[0].text

    else:
        response = openai_client.responses.create(model = model, input = prompt)

        return response.output_text

## ------------------------------------------------------ ##
def generate_chart_code(instruction: str, model: str, out_path_v1: str) -> str:
    prompt = f"""
            You are a data visualization expert.

            Return your answer *strictly* in this format:

            <execute_python>
            # valid python code here
            </execute_python>

            Do not add explanations, only the tags and the code.

                The code should create a visualization from a DataFrame 'df' with these columns:
                - date (M/D/YY)
                - time (HH:MM)
                - cash_type (card or cash)
                - card (string)
                - price (number)
                - coffee_name (string)
                - quarter (1-4)
                - month (1-12)
                - year (YYYY)

                User instruction: {instruction}

                Requirements for the code:
                1. Assume the DataFrame is already loaded as 'df'.
                2. Use matplotlib for plotting.
                3. Add clear title, axis labels, and legend if needed.
                4. Save the figure as '{out_path_v1}' with dpi=300.
                5. Do not call plt.show().
                6. Close all plots with plt.close().

                Return ONLY the code wrapped in <execute_python> tags.
                """

    response = get_response(model, prompt)

    return response

## ------------------------------------------------------ ##
instructions = "Create a chart showing year-over-year Q1 sales by drink type."
model = "gpt-4o-mini"

file_name_version_1 = "chart_v1.png"
code_v1 = generate_chart_code(instructions, model, file_name_version_1)
print_html(code_v1, title = "Generated Code Output (V1)")

## ------------------------------------------------------ ##
match = re.search(r"<execute_python>([\s\S]*?)</execute_python>", code_v1)

if match:
    initial_code = match.group(1).strip()
    exec_globals = {"df" : df, "plt" : plt, "pd" : pd}
    exec(initial_code, exec_globals)

## ------------------------------------------------------ ##
print_html(file_name_version_1, is_image = True, title = "Generated Chart (V1)")

## ------------------------------------------------------ ##
def _anthropic_call_json_with_image(client, model_name: str, prompt: str,
                                    media_type: str, b64: str) -> str:
    msg = client.messages.create(model = model_name,
                                max_tokens = 2000,
                                temperature = 0,
                                system = ("You are a careful assistant. Respond with a single "
                                          "valid JSON object only. Do not include markdown, "
                                          "code fences, or commentary outside JSON."),
                                messages = [{"role" : "user",
                                             "content": [{"type" : "text", "text" : prompt},
                                                         {"type" : "image",
                                                          "source" : {"type" : "base64",
                                                                      "media_type" : media_type,
                                                                      "data": b64}
                                                        }]
                                            }]
                                )

    parts = []

    for block in (msg.content or []):
        if getattr(block, "type", None) == "text":
            parts.append(block.text)

    return "".join(parts).strip()

## ------------------------------------------------------ ##
def reflect_on_image_and_regenerate(chart_path: str, instruction: str, client,
                                    model_name: str, out_path_v2: str) -> tuple[str, str]:
    media_type, b64 = encode_image_b64(chart_path)

    prompt = (
        "You are a data visualization expert. First, critique how well the attached chart "
        "communicates the instruction. Then return improved matplotlib code.\n\n"
        "STRICT OUTPUT FORMAT (JSON only):\n"
        "{\n"
        ' "feedback": "<brief, specific critique and suggestions>",\n'
        ' "refined_code": "<ONLY python code, wrapped in <execute_python> tags; assumes df exists;'
        f'saves to \'{out_path_v2}\' with dpi=300; NO plt.show(); DO call plt.close() at end>"\n'
        "}\n\n"
        "Constraints for the refined code:\n"
        "- Use pandas/matplotlib only (no seaborn).\n"
        "- Assume df exists; no file reads.\n"
        f"- Save to '{out_path_v2}' with dpi=300.\n"
        "- If year/month/quarter are needed and missing, derive them from df['date'] with:\n"
        "  df['date'] = pd.to_datetime(df['date'], errors='coerce')\n"
        "  if 'year' not in df.columns: df['year'] = df['date'].dt.year\n"
        "  if 'month' not in df.columns: df['month'] = df['date'].dt.month\n"
        "  if 'quarter' not in df.columns: df['quarter'] = df['date'].dt.quarter\n\n"
        "Schema (columns you may reference):\n"
        "- date (M/D/YY)\n"
        "- time (HH:MM)\n"
        "- cash_type (card or cash)\n"
        "- card (string)\n"
        "- price (number)\n"
        "- coffee_name (string)\n"
        "- quarter (1-4)\n"
        "- month (1-12)\n"
        "- year (YYYY)\n\n"
        f"Instruction:\n{instruction}\n"
        )

    lower = model_name.lower()

    if "claude" in lower or "anthropic" in lower:
        content = _anthropic_call_json_with_image(client, model_name, prompt, media_type, b64)

    else:
        data_url = f"data: {media_type}; base64, {b64}"
        resp = client.responses.create(model = model_name,
                                       input = [{"role" : "user",
                                                 "content" : [{"type" : "input_text",
                                                               "text" : prompt},
                                                               {"type" : "input_image",
                                                                "image_url" : data_url},
                                                            ],
                                                }],
                                        )
        content = (resp.output_text or "").strip()

    try:
        obj = json.loads(content)

    except Exception:
        m = re.search(r"\{.*\}", content, flags = re.DOTALL)
        obj = json.loads(m.group(0)) if m else {"feedback" : content, "refined_code" : ""}

    feedback = str(obj.get("feedback", "")).strip()
    refined_code = ensure_execute_python_tags(str(obj.get("refined_code", "")).strip())

    return feedback, refined_code

## ------------------------------------------------------ ##
file_name_version_2 = "chart_v2.png"

feedback, code_v2 = reflect_on_image_and_regenerate(chart_path = file_name_version_1,
                                                    instruction = instructions,
                                                    client = openai_client,
                                                    model_name = "gpt-4.1",
                                                    out_path_v2 = file_name_version_2)

## ------------------------------------------------------ ##
print_html(feedback, title = "Feedback on V1 Chart")
print_html(code_v2, title = "Regenerated Code Output (V2)")

## ------------------------------------------------------ ##
match = re.search(r"<execute_python>([\s\S]*?)</execute_python>", code_v2)

if match:
    reflected_code = match.group(1).strip()
    exec_globals = {"df" : df, "plt" : plt, "pd" : pd}
    exec(reflected_code, exec_globals)

## ------------------------------------------------------ ##
print_html(file_name_version_2, is_image = True, title = "Regenerated Chart (V2)")

## ------------------------------------------------------ ##
def run_workflow(dataset_path: str, user_instructions: str, generation_model: str,
                 evaluation_model: str, image_basename: str = "chart", ) -> Dict[str, Any]:
    df = load_and_prepare_data(dataset_path)
    print_html(df.sample(n = 5), title = "Random Sample of Dataset")

    out_v1 = f"{image_basename}_v1.png"
    out_v2 = f"{image_basename}_v2.png"


    print_html("Step 1: Generating chart code... 📈\n")
    code_v1 = generate_chart_code(instruction = user_instructions, model = generation_model,
                                  out_path_v1 = out_v1, )
    print_html(code_v1, title = "Generated Code Output (V1)")


    print_html("Step 2: Executing chart code... 💻\n")
    chart_path_v1 = None
    match = re.search(r"<execute_python>([\s\S]*?)</execute_python>", code_v1)

    if match:
        initial_code = match.group(1).strip()
        exec_globals = {"df" : df, "plt" : plt, "pd" : pd}
        exec(initial_code, exec_globals)

        if os.path.exists(out_v1):
            chart_path_v1 = out_v1

    print_html(chart_path_v1, is_image = True, title = "Generated Chart (V1)")


    print_html("Step 3: Evaluating and refining chart... 🔁\n")
    feedback, code_v2 = reflect_on_image_and_regenerate(chart_path = out_v1,
                                                        instruction = user_instructions,
                                                        client = openai_client,
                                                        model_name = evaluation_model,
                                                        out_path_v2 = out_v2, )
    print_html(feedback, title = "Feedback on V1 Chart")
    print_html(code_v2, title = "Regenerated Code Output (V2)")


    print_html("Step 4: Executing refined chart code... 🖼️\n")
    chart_path_v2 = None
    match = re.search(r"<execute_python>([\s\S]*?)</execute_python>", code_v2)

    if match:
        reflected_code = match.group(1).strip()
        exec_globals = {"df" : df, "plt" : plt, "pd" : pd}
        exec(reflected_code, exec_globals)

        if os.path.exists(out_v2):
            chart_path_v2 = out_v2

    print_html(chart_path_v2, is_image = True, title = "Regenerated Chart (V2)")

    return {"status" : "success", "dataset_path" : dataset_path,
            "user_instructions" : user_instructions, "generation_model" : generation_model,
            "evaluation_model" : evaluation_model, "chart_code_v1" : code_v1,
            "original_chart_path" : chart_path_v1, "feedback" : feedback,
            "refined_code_v2" : code_v2, "refined_chart_path": chart_path_v2, }

## ------------------------------------------------------ ##
result = run_workflow(dataset_path = "coffee_sales.csv",
                      user_instructions = ("Create a chart showing year-over-year Q1 sales by "
                                            "drink type."),
                      generation_model = "gpt-4.1-mini",
                      evaluation_model = "o4-mini",
                      image_basename = "drink_sales")
