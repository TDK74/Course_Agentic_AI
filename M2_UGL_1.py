import json
import re
import utils


## ------------------------------------------------------ ##
df = utils.load_and_prepare_data('coffee_sales.csv')
utils.print_html(df.sample(n = 5), title = "Random Sample of Coffee Sales Data")

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
            7. Add all necessary import python statements

            Return ONLY the code wrapped in <execute_python> tags.
            """

    response = utils.get_response(model, prompt)

    return response

## ------------------------------------------------------ ##
code_v1 = generate_chart_code(instruction = ("Create a plot comparing Q1 coffee sales in 2024 and "
                                            "2025 using the data in coffee_sales.csv."),
                            model = "gpt-4o-mini", out_path_v1 = "chart_v1.png")

utils.print_html(code_v1, title = "LLM output with first draft code")

## ------------------------------------------------------ ##
match = re.search(r"<execute_python>([\s\S]*?)</execute_python>", code_v1)

if match:
    initial_code = match.group(1).strip()
    utils.print_html(initial_code, title = "Extracted Code to Execute")
    exec_globals = {"df" : df}
    exec(initial_code, exec_globals)

utils.print_html(content = "chart_v1.png", title = "Generated Chart (V1)", is_image = True)

## ------------------------------------------------------ ##
def reflect_on_image_and_regenerate(chart_path: str, instruction: str, model_name: str,
                                    out_path_v2: str, code_v1: str, ) -> tuple[str, str]:
    media_type, b64 = utils.encode_image_b64(chart_path)

    prompt = f"""
            You are a data visualization expert.
            Your task: critique the attached chart and the original code against the given
             instruction, then return improved matplotlib code.

            Original code (for context):
            {code_v1}

            OUTPUT FORMAT (STRICT!):
            1) First line: a valid JSON object with ONLY the "feedback" field.
            Example: {{"feedback": "The legend is unclear and the axis labels overlap."}}

            2) After a newline, output ONLY the refined Python code wrapped in:
            <execute_python>
            ...
            </execute_python>

            3) Import all necessary libraries in the code. Don't assume any imports from
              the original code.

            HARD CONSTRAINTS:
            - Do NOT include Markdown, backticks, or any extra prose outside the two parts above.
            - Use pandas/matplotlib only (no seaborn).
            - Assume df already exists; do not read from files.
            - Save to '{out_path_v2}' with dpi=300.
            - Always call plt.close() at the end (no plt.show()).
            - Include all necessary import statements.

            Schema (columns available in df):
            - date (M/D/YY)
            - time (HH:MM)
            - cash_type (card or cash)
            - card (string)
            - price (number)
            - coffee_name (string)
            - quarter (1-4)
            - month (1-12)
            - year (YYYY)

            Instruction:
            {instruction}
            """

    lower = model_name.lower()

    if "claude" in lower or "anthropic" in lower:
        content = utils.image_anthropic_call(model_name, prompt, media_type, b64)

    else:
        content = utils.image_openai_call(model_name, prompt, media_type, b64)

    lines = content.strip().splitlines()
    json_line = lines[0].strip() if lines else ""

    try:
        obj = json.loads(json_line)

    except Exception as e:
        m_json = re.search(r"\{.*?\}", content, flags = re.DOTALL)

        if m_json:
            try:
                obj = json.loads(m_json.group(0))

            except Exception as e2:
                obj = {"feedback": f"Failed to parse JSON: {e2}", "refined_code": ""}

        else:
            obj = {"feedback": f"Failed to find JSON: {e}", "refined_code": ""}

    m_code = re.search(r"<execute_python>([\s\S]*?)</execute_python>", content)
    refined_code_body = m_code.group(1).strip() if m_code else ""
    refined_code = utils.ensure_execute_python_tags(refined_code_body)

    feedback = str(obj.get("feedback", "")).strip()

    return feedback, refined_code

## ------------------------------------------------------ ##
feedback, code_v2 = reflect_on_image_and_regenerate(chart_path = "chart_v1.png",
                                                    instruction = ("Create a plot comparing Q1 "
                                                                    "coffee sales in 2024 and 2025 "
                                                                    "using the data in "
                                                                    "coffee_sales.csv."),
                                                    model_name = "o4-mini",
                                                    out_path_v2 = "chart_v2.png",
                                                    code_v1 = code_v1, )

utils.print_html(feedback, title  ="Feedback on V1 Chart")
utils.print_html(code_v2, title = "Regenerated Code Output (V2)")

## ------------------------------------------------------ ##
match = re.search(r"<execute_python>([\s\S]*?)</execute_python>", code_v2)

if match:
    reflected_code = match.group(1).strip()
    exec_globals = {"df": df}
    exec(reflected_code, exec_globals)

utils.print_html(content = "chart_v2.png", title = "Regenerated Chart (V2)", is_image = True)

## ------------------------------------------------------ ##
def run_workflow(dataset_path: str, user_instructions: str, generation_model: str,
                reflection_model: str, image_basename: str = "chart", ):
    df = utils.load_and_prepare_data(dataset_path)
    utils.print_html(df.sample(n = 5), title = "Random Sample of Dataset")

    out_v1 = f"{image_basename}_v1.png"
    out_v2 = f"{image_basename}_v2.png"

    utils.print_html("Step 1: Generating chart code (V1)… 📈")
    code_v1 = generate_chart_code(instruction = user_instructions, model = generation_model,
                                out_path_v1 = out_v1, )
    utils.print_html(code_v1, title = "LLM output with first draft code (V1)")

    utils.print_html("Step 2: Executing chart code (V1)… 💻")
    match = re.search(r"<execute_python>([\s\S]*?)</execute_python>", code_v1)

    if match:
        initial_code = match.group(1).strip()
        exec_globals = {"df" : df}
        exec(initial_code, exec_globals)

    utils.print_html(out_v1, is_image = True, title = "Generated Chart (V1)")

    utils.print_html("Step 3: Reflecting on V1 (image + code) and generating improvements… 🔁")
    feedback, code_v2 = reflect_on_image_and_regenerate(chart_path = out_v1,
                                                        instruction = user_instructions,
                                                        model_name = reflection_model,
                                                        out_path_v2 = out_v2, code_v1 = code_v1, )

    utils.print_html(feedback, title = "Reflection feedback on V1")
    utils.print_html(code_v2, title = "LLM output with revised code (V2)")

    utils.print_html("Step 4: Executing refined chart code (V2)… 🖼️")
    match = re.search(r"<execute_python>([\s\S]*?)</execute_python>", code_v2)

    if match:
        reflected_code = match.group(1).strip()
        exec_globals = {"df" : df}
        exec(reflected_code, exec_globals)

    utils.print_html(out_v2, is_image = True, title = "Regenerated Chart (V2)")

    return {"code_v1" : code_v1, "chart_v1" : out_v1, "feedback" : feedback, "code_v2" : code_v2,
            "chart_v2" : out_v2, }

## ------------------------------------------------------ ##
user_instructions = ("Create a plot comparing Q1 coffee sales in 2024 and 2025 using the data in "
                    "coffee_sales.csv.")
generation_model = "gpt-4.1-mini"
reflection_model = "o4-mini"
image_basename = "drink_sales"

_ = run_workflow(dataset_path = "coffee_sales.csv", user_instructions = user_instructions,
                generation_model = generation_model, reflection_model = reflection_model,
                image_basename = image_basename)
