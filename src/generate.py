import os
import json
import time
from mistralai import Mistral
from groq import Groq

MISTRAL_API_KEY= "MISTRAL"
GROQ_API_KEY= "GROQ"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

mistral_client = Mistral(api_key=MISTRAL_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)


SCENARIOS = [
    {
        "id": 1,
        "intent": "Follow up after client meeting",
        "facts": ["Met Jan 12", "Discussed Q2 roadmap", "Next step: SOW by Jan 20", "Client: Apex Solutions"],
        "tone": "formal"
    },
    {
        "id": 2,
        "intent": "Request project deadline extension",
        "facts": ["Original deadline: Feb 1", "Need 2 extra weeks", "Blocker: vendor API delay", "Project: DataSync v2"],
        "tone": "urgent"
    },
    {
        "id": 3,
        "intent": "Introduce a new team member",
        "facts": ["Name: Priya Sharma", "Role: Lead ML Engineer", "Starts Monday", "Previously at Google DeepMind"],
        "tone": "warm and casual"
    },
    {
        "id": 4,
        "intent": "Decline a vendor proposal",
        "facts": ["Evaluated 3 vendors", "Budget constraint: $50K", "May revisit in Q4", "Vendor: TechBridge Inc."],
        "tone": "diplomatic"
    },
    {
        "id": 5,
        "intent": "Request invoice payment",
        "facts": ["Invoice #4421", "Due date passed 15 days ago", "Amount: $12,400", "Client: Meridian Corp"],
        "tone": "assertive"
    },
    {
        "id": 6,
        "intent": "Announce product launch delay",
        "facts": ["Launch moved to March 15", "Reason: QA findings", "Compensation: extended beta access", "Product: FlowApp 2.0"],
        "tone": "empathetic"
    },
    {
        "id": 7,
        "intent": "Schedule a performance review",
        "facts": ["Review week of Feb 10", "Duration: 45 minutes", "Focus: OKR attainment", "Employee: James Liu"],
        "tone": "professional"
    },
    {
        "id": 8,
        "intent": "Thank a client after contract signing",
        "facts": ["Contract value: $250K", "18-month engagement", "Kickoff call: Feb 3", "Client: Stratford Group"],
        "tone": "warm formal"
    },
    {
        "id": 9,
        "intent": "Escalate a critical production bug",
        "facts": ["System down since 2 AM EST", "3,000 users affected", "ETA unknown", "Service: PayGate API"],
        "tone": "urgent"
    },
    {
        "id": 10,
        "intent": "Propose a partnership",
        "facts": ["Overlap in EdTech", "Rev-share model", "30-min call", "Company: LearnForge"],
        "tone": "persuasive"
    }
]


SYSTEM_PROMPT = """<|role|>
You are an expert professional email writer.

<|task|>
Generate high-quality, professional emails using given intent, facts, and tone.

<|requirements|>
- Include all facts naturally
- Match tone exactly
- Use proper email structure (subject, greeting, body, closing)
- Be clear, concise, and professional
"""

def build_prompt(s):

    """
    Assistant function
    Input: 's' is a dictionary which contains "intent", "facts" (key facts) and "tone
    output: Prompt
    """
    

    return f"""
<|instructions|>
Generate a professional email based on:
- Intent
- Facts
- Tone

Follow examples carefully.

<|examples|>

Example 1:
Intent: Follow up after meeting
Facts:
- Met Jan 10
- Discussed roadmap
- Next step: proposal
Tone: formal

Output:
Subject: Follow-up on Meeting

Dear Sir/Madam,

Thank you for meeting on Jan 10 to discuss the roadmap. As discussed, we will share the proposal as the next step.

Best regards,
Boddu Sri Pavan


Example 2:
Intent: Request deadline extension
Facts:
- Deadline Feb 1
- Need 1 week extension
- Reason: API delay
Tone: urgent

Output:
Subject: Urgent: Deadline Extension Request

Hi,

Due to an API delay, we request a one-week extension beyond the Feb 1 deadline. We are actively working to resolve this.

Please confirm.

Thanks,
Boddu Sri Pavan


<|input|>
Intent: {s['intent']}

Facts:
{chr(10).join(f"- {f}" for f in s['facts'])}

Tone: {s['tone']}

<|output|>
"""


def mistral_generate(prompt):
    res = mistral_client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    return res.choices[0].message.content.strip()

def groq_generate(prompt):
    res = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    return res.choices[0].message.content.strip()



def main():
    outputs = []

    for s in SCENARIOS:
        print(f"Scenario {s['id']}")

        prompt = build_prompt(s)

        mistral_email = mistral_generate(prompt)
        groq_email = groq_generate(prompt)

        outputs.append({
            "id": s["id"],
            "intent": s["intent"],
            "facts": s["facts"],
            "tone": s["tone"],
            "mistral_output": mistral_email,
            "gpt_output": groq_email
        })

        # time.sleep(1)
        time.sleep(1)

    with open( os.path.join(OUTPUT_DIR, "generated_outputs.json") , "w") as f:
        json.dump(outputs, f, indent=2)


if __name__ == "__main__":
    main()