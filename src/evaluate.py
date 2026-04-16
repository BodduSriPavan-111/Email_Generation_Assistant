import os
import json
import csv
from mistralai import Mistral
from groq import Groq
import re


MISTRAL_API_KEY= "MISTRAL"
GROQ_API_KEY= "GROQ"

client = Groq(api_key=GROQ_API_KEY)
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

INPUT_PATH = os.path.join(OUTPUT_DIR, "generated_outputs.json")
CSV_PATH = os.path.join(OUTPUT_DIR, "qwen_comparison.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "qwen_final_summary.json")


def mistral_judge(prompt):
    res = mistral_client.chat.complete(
        model= "mistral-medium-latest", #"mistral-large-latest",
        messages=[
            {"role": "system", "content": "You are a strict evaluator. Always return valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    text= res.choices[0].message.content.strip()
    # print( text)

    text = re.sub(r"```.*?\n", "", text)
    text = text.replace("```", "").strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:

        # print( match.group(0) )
        return match.group(0)

    return text


def qwen_judge(prompt):
    res = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    text= res.choices[0].message.content.strip()

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    
    text = re.sub(r"```.*?\n", "", text)
    text = text.replace("```", "")

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        json_str = match.group(0)

        print( json_str)
        return json_str
   
    return res.choices[0].message.content.strip()


def fact_recall(email, facts):
    
    prompt = f"""
You are a strict evaluator.

Facts:
{facts}

Email:
{email}

Instructions:
- Check each fact carefully
- Give 1 only if clearly present or strongly implied
- Otherwise 0

Return EXACT JSON:
{{"scores":[1,0,1,...]}}
NO explanation.
"""
    try:
        data = json.loads(qwen_judge(prompt))
        return sum(data["scores"]) / len(facts)
    except Exception as e:
        print(str(e))
        return 0.5


def tone_score(email, tone):
    prompt = f"""
Expected Tone: {tone}

Email:
{email}

Rate tone match (1 to 5 scale)

Return JSON:
{{"score":4}}
"""
    try:
        return json.loads(qwen_judge(prompt))["score"] / 5
    except Exception as e:
        print(str(e))
        return 0.5


def fluency_score(email):
    prompt = f"""
Rate grammar & fluency (1 to 5 scale):

{email}

Return JSON:
{{"score":5}}
"""
    try:
        return json.loads(qwen_judge(prompt))["score"] / 5
    except Exception as e:
        print(str(e))
        return 0.5


def evaluate(email, facts, tone):
    fr = fact_recall(email, facts)
    ts = tone_score(email, tone)
    fs = fluency_score(email)

    composite = (fr + ts + fs) / 3

    return fr, ts, fs, composite



def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError("Run generate_emails.py first")

    with open(INPUT_PATH) as f:
        data = json.load(f)

    results = []

    mistral_scores = []
    gpt_scores = []

    for item in data:
        m = evaluate(item["mistral_output"], item["facts"], item["tone"])
        g = evaluate(item["gpt_output"], item["facts"], item["tone"])

        mistral_scores.append(m[3])
        gpt_scores.append(g[3])

        results.append({
            "id": item["id"],
            "mistral_score": round(m[3], 4),
            "gpt_score": round(g[3], 4),
            "better_model": "mistral" if m[3] > g[3] else "gpt"
        })

        print(f"Scenario {item['id']} → Mistral: {m[3]:.2f}, GPT: {g[3]:.2f}")

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    avg_m = sum(mistral_scores) / len(mistral_scores)
    avg_g = sum(gpt_scores) / len(gpt_scores)

    best_model = "mistral" if avg_m > avg_g else "gpt"

    print("FINAL REPORT")
    print(f"Mistral Avg Score (mistral-small-latest) : {avg_m:.4f}")
    print(f"GPT Avg Score     (openai/gpt-oss-120b)  :    {avg_g:.4f}")
    print(f"BEST MODEL: {best_model.upper()}")

    summary = {
        "mistral_avg": round(avg_m, 4),
        "gpt_avg": round(avg_g, 4),
        "best_model": best_model,
        "metric_definitions": {
            "Fact Recall": "Measures inclusion of all input facts",
            "Tone Score": "Measures alignment with requested tone",
            "Fluency Score": "Measures grammar and readability",
            "Composite": "Average of all three metrics"
        }
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()