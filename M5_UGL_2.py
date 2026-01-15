import base64
import json
import os
import re
import aisuite
import openai
import requests
import tools
import utils

from datetime import datetime
from io import BytesIO
from dotenv import load_dotenv
from IPython.display import Markdown, display
from PIL import Image


load_dotenv()
client = aisuite.Client()

## ------------------------------------------------------ ##
tools.tavily_search_tool('trends in sunglasses fashion')

## ------------------------------------------------------ ##
tools.product_catalog_tool()

## ------------------------------------------------------ ##
def market_research_agent(return_messages: bool = False):
    utils.log_agent_title_html("Market Research Agent", "🕵️‍♂️")

    prompt_ = f"""
            You are a fashion market research agent tasked with preparing a trend analysis for a \
            summer sunglasses campaign.

            Your goal:
            1. Explore current fashion trends related to sunglasses using web search.
            2. Review the internal product catalog to identify items that align with those trends.
            3. Recommend one or more products from the catalog that best match emerging trends.
            4. If needed, today date is {datetime.now().strftime("%Y-%m-%d")}.

            You can call the following tools:
            - tavily_search_tool: to discover external web trends.
            - product_catalog_tool: to inspect the internal sunglasses catalog.

            Once your analysis is complete, summarize:
            - The top 2-3 trends you found.
            - The product(s) from the catalog that fit these trends.
            - A justification of why they are a good fit for the summer campaign.
            """

    messages = [{"role" : "user", "content" : prompt_}]
    tools_ = tools.get_available_tools()

    while True:
        response = client.chat.completions.create(model = "openai:o4-mini", messages = messages,
                                                tools = tools_, tool_choice = "auto")

        msg = response.choices[0].message

        if msg.content:
            utils.log_final_summary_html(msg.content)

            return (msg.content, messages) if return_messages else msg.content

        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                utils.log_tool_call_html(tool_call.function.name, tool_call.function.arguments)
                result = tools.handle_tool_call(tool_call)
                utils.log_tool_result_html(result)

                messages.append(msg)
                messages.append(tools.create_tool_response_message(tool_call, result))

        else:
            utils.log_unexpected_html()

            return (("[⚠️ Unexpected: No tool_calls or content returned]", messages)
                    if return_messages else "[⚠️ Unexpected: No tool_calls or content returned]")

## ------------------------------------------------------ ##
market_research_result = market_research_agent()

## ------------------------------------------------------ ##
def graphic_designer_agent(trend_insights: str, caption_style: str = "short punchy",
                            size: str = "1024x1024") -> dict:
    utils.log_agent_title_html("Graphic Designer Agent", "🎨")

    system_message = (
                    "You are a visual marketing assistant. Based on the input trend insights, "
                    "write a creative and visual prompt for an AI image generation model, "
                    "and also a short caption."
                    )

    user_prompt = f"""
                Trend insights:
                {trend_insights}

                Please output:
                1. A vivid, descriptive prompt to guide image generation.
                2. A marketing caption in style: {caption_style}.

                Respond in this format:
                {{"prompt": "...", "caption": "..."}}
                """

    chat_response = client.chat.completions.create(model = "openai:o4-mini",
                                                    messages = [{"role" : "system",
                                                                "content" : system_message},
                                                                {"role" : "user",
                                                                "content" : user_prompt}]
                                                    )

    content = chat_response.choices[0].message.content.strip()
    match = re.search(r'\{.*\}', content, re.DOTALL)
    parsed = json.loads(match.group(0)) if match else {"error": "No JSON returned", "raw": content}

    prompt = parsed["prompt"]
    caption = parsed["caption"]

    openai_client = openai.OpenAI()

    image_response = openai_client.images.generate(model = "dall-e-3", prompt = prompt, size = size,
                                            quality = "standard", n = 1, response_format = "url")

    image_url = image_response.data[0].url

    img_bytes = requests.get(image_url).content
    img = Image.open(BytesIO(img_bytes))

    filename = os.path.basename(image_url.split("?")[0])
    image_path = filename
    img.save(image_path)

    utils.log_final_summary_html(f"""
                                <h3>Generated Image and Caption</h3>

                                <p><strong>Image Path:</strong> <code>{image_path}</code></p>

                                <p><strong>Generated Image:</strong></p>
                                <img src="{image_path}" alt="Generated Image" \
                                    style="max-width: 100%; height: auto; border: 1px solid #ccc; \
                                    border-radius: 8px; margin-top: 10px; margin-bottom: 10px;">

                                <p><strong>Prompt:</strong> {prompt}</p>
                                """)

    return {"image_url" : image_url, "prompt" : prompt, "caption" : caption,
            "image_path" : image_path}

## ------------------------------------------------------ ##
graphic_designer_agent_result = graphic_designer_agent( trend_insights = market_research_result, )

## ------------------------------------------------------ ##
def copywriter_agent(image_path: str, trend_summary: str, model: str = "openai:o4-mini") -> dict:
    utils.log_agent_title_html("Copywriter Agent", "✍️")

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    b64_img = base64.b64encode(img_bytes).decode("utf-8")

    messages = [{"role" : "system",
                "content" : ("You are a copywriter that creates elegant campaign quotes "
                             "based on an image and a marketing trend summary.")},
                {"role" : "user",
                "content" : [{"type" : "image_url",
                            "image_url" : {"url" : f"data:image/png;base64,{b64_img}",
                                            "detail" : "auto"}
                            },
                            {"type" : "text",
                            "text" : f"""
                                    Here is a visual marketing image and a trend analysis:

                                    Trend summary:
                                    \"\"\"{trend_summary}\"\"\"

                                    Please return a JSON object like:
                                    {{"quote": "A short, elegant campaign phrase (max 12 words)",
                                      "justification": "Why this quote matches the image and trend"
                                    }}"""
                            }]
                }]

    response = client.chat.completions.create(model = model, messages = messages, )

    content = response.choices[0].message.content.strip()

    utils.log_final_summary_html(content)

    try:
        match = re.search(r'\{.*\}', content, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {"error" : "No valid JSON returned"}

    except Exception as e:
        parsed = {"error" : f"Failed to parse: {e}", "raw" : content}

    parsed["image_path"] = image_path

    return parsed

## ------------------------------------------------------ ##
copywriter_agent_result = copywriter_agent(image_path = graphic_designer_agent_result["image_path"],
                                            trend_summary = market_research_result, )

## ------------------------------------------------------ ##
def packaging_agent(trend_summary: str, image_url: str, quote: str, justification: str,
                    output_path: str = "campaign_summary.md") -> str:
    utils.log_agent_title_html("Packaging Agent", "📦")

    styled_image_html = f"""
                        ![Open the generated file to see]({image_url})
                        """

    beautified_summary = client.chat.completions.create(
                                    model = "openai:o4-mini",
                                    messages = [{"role": "system",
                                                "content": "You are a marketing communication \
                                                            expert writing elegant campaign \
                                                            summaries for executives."},
                                                {"role": "user",
                                                "content": f"""
                                                            Please rewrite the following trend \
                                                            summary to be clear, professional, \
                                                            and engaging for a CEO audience:

                                                            \"\"\"{trend_summary.strip()}\"\"\"
                                                            """
                                                }]
                                    ).choices[0].message.content.strip()

    utils.log_tool_result_html(beautified_summary)

    markdown_content = f"""# 🕶️ Summer Sunglasses Campaign - Executive Summary

                        ## 📊 Refined Trend Insights
                        {beautified_summary}

                        ## 🎯 Campaign Visual
                        {styled_image_html}

                        ## ✍️ Campaign Quote
                        {quote.strip()}

                        ## ✅ Why This Works
                        {justification.strip()}

                        ---

                        *Report generated on {datetime.now().strftime('%Y-%m-%d')}*
                        """

    with open(output_path, "w", encoding = "utf-8") as f:
        f.write(markdown_content)

    return output_path

## ------------------------------------------------------ ##
packaging_agent_result = packaging_agent(trend_summary = market_research_result,
                                        image_url = graphic_designer_agent_result["image_path"],
                                        quote = copywriter_agent_result["quote"],
                                        justification = copywriter_agent_result["justification"],
                                        output_path = (f"campaign_summary_"
                                            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md")
                                        )

## ------------------------------------------------------ ##
with open(packaging_agent_result, "r", encoding = "utf-8") as f:
    md_content = f.read()

display(Markdown(md_content))

## ------------------------------------------------------ ##
def run_sunglasses_campaign_pipeline(output_path: str = "campaign_summary.md") -> dict:
    trend_summary = market_research_agent()
    print("✅ Market research completed")

    visual_result = graphic_designer_agent(trend_insights = trend_summary)
    image_path = visual_result["image_path"]
    print("🖼️ Image generated")

    quote_result = copywriter_agent(image_path = image_path, trend_summary = trend_summary)
    quote = quote_result.get("quote", "")
    justification = quote_result.get("justification", "")
    print("💬 Quote created")

    md_path = packaging_agent(trend_summary = trend_summary, image_url = image_path, quote = quote,
                            justification = justification,
                            output_path = (f"campaign_summary_"
                                            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md")
                            )

    print(f"📦 Report generated: {md_path}")

    return {"trend_summary" : trend_summary, "visual" : visual_result,
            "quote" : quote_result, "markdown_path" : md_path}

## ------------------------------------------------------ ##
results = run_sunglasses_campaign_pipeline()

## ------------------------------------------------------ ##
with open(results["markdown_path"], "r", encoding = "utf-8") as f:
    md_content = f.read()

display(Markdown(md_content))
