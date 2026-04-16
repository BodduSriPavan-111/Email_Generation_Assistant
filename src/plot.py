import os
import json
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

mistral_path = os.path.join(OUTPUT_DIR, "mistral_final_summary.json")
qwen_path = os.path.join(OUTPUT_DIR, "qwen_final_summary.json")

with open(mistral_path) as f:
    mistral_data = json.load(f)

with open(qwen_path) as f:
    qwen_data = json.load(f)


mistral_judge_mistral = mistral_data["mistral_avg"]
mistral_judge_gpt = mistral_data["gpt_avg"]

qwen_judge_mistral = qwen_data["mistral_avg"]
qwen_judge_gpt = qwen_data["gpt_avg"]

final_mistral = (mistral_judge_mistral + qwen_judge_mistral) / 2
final_gpt = (mistral_judge_gpt + qwen_judge_gpt) / 2

models = ["Mistral Model", "GPT Model"]
scores = [final_mistral, final_gpt]

plt.figure()
plt.bar(models, scores, color=["darkorange", "cornflowerblue"])
plt.xlabel("Models")
plt.ylabel("Aggregated Score")
plt.title("Final Model Comparison (Across Judges)")

plot_path = os.path.join(OUTPUT_DIR, "final_comparison.png")
plt.savefig(plot_path)

best_model = "Mistral" if final_mistral > final_gpt else "GPT"

print("FINAL AGGREGATED REPORT")
print(f"Mistral Final Score: {final_mistral:.4f}")
print(f"GPT Final Score:     {final_gpt:.4f}")
print(f"BEST MODEL: {best_model}")