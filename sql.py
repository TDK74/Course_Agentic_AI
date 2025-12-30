import aisuite as ais
import pandas as pd

from dotenv import load_dotenv
from typing import Any, Dict, Tuple
from utils import create_transactions_db, get_schema, execute_sql, print_html


_ = load_dotenv()

## ------------------------------------------------------ ##
client = ais.Client()

## ------------------------------------------------------ ##
create_transactions_db()

## ------------------------------------------------------ ##
print_html(get_schema('products.db'))

## ------------------------------------------------------ ##
def generate_sql(question: str, schema: str, model: str = "openai:gpt-3.5-turbo") -> str:
    prompt = f"""
            You are a SQL assistant. Given the schema and the user's question, write a SQL \
            query for SQLite.

            Schema:
            {schema}

            User question:
            {question}

            Respond with the SQL only.
            """
    response = client.chat.completions.create(model = model,
                                              messaages = [{"role" : "user", "content" : prompt}],
                                              temperature = 0, )

    return response.choices[0].message.content.strip()

## ------------------------------------------------------ ##
def evaluate_and_refine_sql(question: str, sql_query: str, df: pd.DataFrame, schema: str,
                            model: str = "openai:gpt-4o", ) -> Tuple[str, str]:
    prompt = f"""
            You are a SQL reviewer and refiner.

            User asked:
            {question}

            Original SQL:
            {sql_query}

            SQL Output:
            {df.to_markdown(index=False)}

            Table Schema:
            {schema}

            Step 1: Briefly evaluate if the SQL output answers the user's question.
            Step 2: If the SQL could be improved, provide a refined SQL query.
            If the original SQL is already correct, return it unchanged.

            Return a strict JSON object with two fields:
            - "feedback": brief evaluation and suggestions
            - "refined_sql": the final SQL to run
            """
    response = client.chat.completions.create(model = model,
                                              messages = [{"role" : "user", "content" : prompt}],
                                              temperature = 0, )

    import json


    content = response.choices[0].message.content

    try:
        obj = json.loads(content)
        feedback = str(obj.get("feedback", "")).strip()
        refined_sql = str(obj.get("refined_sql", sql_query)).strip()

        if not refined_sql:
            refined_sql = sql_query

    except Exception:
        feedback = content.strip()
        refined_sql = sql_query

    return feedback, refined_sql

## ------------------------------------------------------ ##
def run_sql_workflow(db_path: str, question: str, model_generation: str = "openai:gpt-4.1",
                     model_evaluation: str = "openai:gpt-4.1") -> Dict[str, Any]:
    schema = get_schema(db_path)
    print_html("📘 Get schema: \n" + schema)

    sql = generate_sql(question, schema, model_generation)
    print_html("🧠 Generate SQL (V1): \n" + sql)

    df = execute_sql(sql, db_path)
    print_html("📊 RExecute V1 query → Output: \n" + df.to_html())

    feedback, refined_sql = evaluate_and_refine_sql(question = question,
                                                    sql_query = sql,
                                                    df = df,
                                                    schema = schema,
                                                    model = model_evaluation, )

    print_html("📝 Reflect on V1 SQL/output: \n" + feedback)
    print_html("🔁 Write V2 query: \n" + refined_sql)

    refined_df = execute_sql(refined_sql, db_path)
    print_html("✅ Execute V2 query → Final answer: \n" + refined_df.to_html())

    return {"original_sql" : sql,
            "refined_sql" : refined_sql,
            "original_results" : df,
            "refined_results" : refined_df,
            "feedback" : feedback, }

## ------------------------------------------------------ ##
results = run_sql_workflow("product.db",
                           "Which color of product has the highest total sales?",
                           model_generation = "openai:gpt-4.1",
                           model_evaluation = "openai:gpt-4.1")
