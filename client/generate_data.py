# client/generate_data.py

import json
import os

# Use environment variable or default to relative path for local execution
CONFIG_PATH = os.getenv("CONFIG_DIR", "./configs")
WORKLOAD_FILE = os.path.join(CONFIG_PATH, "workloads.json")

os.makedirs(CONFIG_PATH, exist_ok=True)

# -------------------------------------------------------------------------
# PART 1: CLINICAL CONTEXT & TRANSCRIPT (The "Heavy" Payload)
# -------------------------------------------------------------------------

PATIENT_CONTEXT = """
PATIENT RECORD: Robert "Bob" Vance | DOB: 12-04-1958 | ID: #99283
ALLERGIES: Penicillin (Anaphylaxis), Latex (Mild rash).

ACTIVE PROBLEM LIST:
1. Type 2 Diabetes Mellitus (uncontrolled, A1C 8.9%).
2. COPD (Chronic Obstructive Pulmonary Disease) - Gold Stage II.
3. Hypertension - benign essential.
4. Chronic Lower Back Pain (L4-L5 degeneration).

MEDICATION LIST (Active):
- Metformin 1000mg BID (Twice daily).
- Lantus Solostar 20 units SQ at bedtime.
- Lisinopril 20mg daily.
- Spiriva Respimat 2.5mcg daily.
- Albuterol HFA 90mcg (PRN).
- Gabapentin 300mg TID.

RECENT LABS (2 days ago):
- HbA1c: 9.1% (High - Flagged).
- eGFR: 58 mL/min (Mildly reduced).
- LDL Cholesterol: 145 (High).
"""

COMPLEX_TRANSCRIPT = """
Doctor: Morning, Bob. Good to see you. How are you holding up?
Patient: Oh, you know, Doc. Hanging in there. The drive over was terrible, traffic on the 405 is a nightmare.
Doctor: I heard. Well, I'm glad you made it. We have a lot to cover today. I saw your lab results come in yesterday.
Patient: Yeah, the A1C thing? I saw it on the portal. It's up, isn't it?
Doctor: Unfortunately, yes. It's up to 9.1%. Last time we were at 8.9%, so we're moving in the wrong direction. Talk to me about the insulin. Are you taking the 20 units every night?
Patient: Well, that's the thing. I was taking it faithfully for the first month. But then my wife got sick—she had that knee surgery I told you about—and everything got chaotic. I think I missed about a week's worth in October. And honestly, Doc, I hate the needles. Even the pen. It stings.
Doctor: I understand, Bob. But at 9.1%, we are risking nerve damage and your kidneys are already showing slight strain. Your eGFR is 58. That's not terrible, but we want to protect what you have left. If the needle is the issue, are you rotating the injection sites? Stomach, thighs?
Patient: Mostly just the stomach. It's easiest.
Doctor: Okay. If you stay in one spot, it gets tough and painful. I want you to try the thigh tonight. But we also need to address the diet. With your wife recovering, who is cooking?
Patient: That would be McDonald's and the pizza place down the street. I know, I know. It's killing me. But I can't stand long enough to cook a full meal because of my back.
Doctor: Let's talk about the back. You're on Gabapentin 300mg three times a day. Is that touching the pain?
Patient: Barely. It takes the edge off the burning, but the ache? The deep ache? It's still a 7 out of 10 most days. I was hoping maybe we could try something stronger? My neighbor gave me one of his... I think it was Percocet? It worked like a charm.
Doctor: Bob, I can't prescribe Percocet for chronic back pain. We have to be very careful with opioids, especially with your COPD. It suppresses breathing.
Patient: But I can't sleep, Doc!
Doctor: I hear you. But taking unprescribed medication is dangerous. Please don't do that again. Instead of opioids, I want to try switching the Gabapentin to Lyrica. It works similarly but some people find it more effective. We can start at 75mg twice a day.
Patient: Lyrica? Is that the one on TV?
Doctor: That's the one. Will you try it?
Patient: I guess so. If it helps me stand longer, maybe I can cook something healthy and fix the diabetes issue. It's all connected, isn't it?
Doctor: It exactly is. Now, speaking of breathing... how is the COPD?
Patient: It's been acting up. Especially in the mornings. I'm coughing up a lot of white stuff.
Doctor: Any fever? Any color in the phlegm? Green or yellow?
Patient: No, just white or clear. But I'm using the rescue inhaler—the Albuterol—like four times a day.
Doctor: Four times is too much. That tells me the maintenance inhaler isn't doing its job, or... are you still using the Spiriva?
Patient: The Spiriva... that's the gray one? Yeah, I take it when I remember. Maybe three times a week?
Doctor: Bob, Spiriva is a daily medication. It opens the airways to prevent the attack. Albuterol is for when the attack happens. You have to take the Spiriva every single morning.
Patient: It's expensive, Doc. The copay went up to $60.
Doctor: Okay, that I can help with. I can switch you to Incruse Ellipta. It's a similar drug, but it might be preferred on your plan. Let me check... yes, I can do that. Will you take it if the price is lower?
Patient: Yeah, $60 is just too much right now.
Doctor: Done. I'll send that in. Now, I noticed something else in your chart. Your LDL cholesterol is 145. We talked about a statin last time.
Patient: The muscle pain pills? No way. My back hurts enough.
Doctor: I know you're worried. But with diabetes and hypertension, your stroke risk is significant. What if we tried a very low dose of Rosuvastatin? It tends to be gentler on the muscles than the one you read about. Just 5mg.
Patient: If I feel even a twinge of muscle pain, I'm stopping.
Doctor: That's a deal. Stop immediately if you feel pain. But give it a shot.
Patient: Alright. 
Doctor: One last thing. I'm doing a depression screen on everyone today. Over the last two weeks, have you felt down, depressed, or hopeless?
Patient: To be honest... watching my wife struggle with the knee, and me not being able to help because of my back... yeah. I feel pretty useless. I don't enjoy watching the football games anymore. I just want to sleep.
Doctor: Thank you for telling me that. That sounds tough. On a scale of 0 to 3, how often do you feel you're a failure or have let yourself down?
Patient: Probably a 2. More than half the days.
Doctor: Okay. I think we should address this. Pain and depression feed off each other. Would you be open to talking to a counselor? We have a social worker here, Sarah, she's great.
Patient: I don't know about talking. Can't I just take a pill?
Doctor: We could add an antidepressant, but we're already changing your back meds and adding the cholesterol med. I'd prefer not to change too many chemicals at once. Let's try one visit with Sarah. Just one. If you hate it, we don't go back.
Patient: Just one visit?
Doctor: Just one.
Patient: Okay. Set it up.
Doctor: Summary time. 
1. We are switching Gabapentin to Lyrica 75mg BID for the back.
2. We are switching Spiriva to Incruse Ellipta for the COPD/Cost issues. Daily use, Bob!
3. We are starting Rosuvastatin 5mg for the cholesterol.
4. I want you to inject the insulin in the thighs, not the stomach, and get back on track.
5. Referral to Sarah for the mood.
I want to see you back in 4 weeks, not 3 months. We need to see if that A1C moves.
Patient: 4 weeks. Got it. Thanks, Doc. I'll try to stay away from the pizza.
Doctor: That's all I ask. Take care, Bob.
"""

CLINICAL_SYSTEM_PROMPT = """
You are an expert AI Medical Scribe.
Task: Generate a professional SOAP note based on the Patient Record and the Consultation Transcript.
Rules:
1. **S**ubjective: Extract HPI, ROS, and Social History. Quote patient for key symptoms.
2. **O**bjective: Use the Vitals and Labs provided in the Record.
3. **A**ssessment: Specific diagnoses. Note "Uncontrolled" status where applicable.
4. **P**lan: Bulleted list of ALL medication changes (Start/Stop/Switch), Referrals, and Follow-up.
5. **Medication Reconciliation**: Explicitly state which drugs were stopped and which were started.
Output Format: JSON.
"""

# -------------------------------------------------------------------------
# PART 2: DIARIZATION SAMPLE (The Logic Test)
# -------------------------------------------------------------------------

DIARIZATION_SAMPLE = """
00:00 [Speaker A]: So, about the pain...
00:02 [Speaker B]: It hurts here.
00:03 [Speaker A]: On the left side?
00:04 [Unknown-1]: Yes.
00:05 [Speaker A]: Okay, take a deep breath.
00:08 [Unknown-2]: Doctor, the X-rays are ready.
00:10 [Speaker A]: Thanks, just put them on the desk.
00:12 [Speaker B]: Do I need to undress?
"""

DIARIZATION_SYSTEM_PROMPT = """
You are a conversation analysis engine.

Input:
A transcript containing one or more segments labeled as [Unknown-1], [Unknown-2], etc.

Task:
For EACH Unknown-X segment, independently infer the most likely speaker identity
based on local context and turn-taking logic.

Rules (must follow):
- Treat each Unknown-X as a separate speaker instance.
- Do NOT assume Unknown segments belong to the same person.
- Classify each Unknown-X independently.
- Each Unknown-X must appear exactly once in the output.
- Do NOT merge Unknowns.
- Do NOT include reasoning or explanations.

Speaker labels:
- Speaker A — Doctor
- Speaker B — Patient
- Speaker C — Nurse / Assistant
- Speaker D — Companion (guardian, parent, partner, child)

Output format (JSON only):
{
  "segments": [
    { "unknown": "Unknown-1", "timestamp": "...", "speaker": "..." },
    { "unknown": "Unknown-2", "timestamp": "...", "speaker": "..." }
  ]
}
"""

def generate_configs():
    clinical_input = PATIENT_CONTEXT + "\n\n=== TRANSCRIPT ===\n" + COMPLEX_TRANSCRIPT
    
    workloads = {
        "diarization": {
            "system_prompt": DIARIZATION_SYSTEM_PROMPT,
            "user_prompt": DIARIZATION_SAMPLE,
            "max_tokens": 1024,
            "temperature": 0.0,
            "concurrency": 64,
            "requests_count": 200
        },
        "clinical": {
            "system_prompt": CLINICAL_SYSTEM_PROMPT,
            "user_prompt": clinical_input,
            "max_tokens": 1500,
            "temperature": 0.1,
            "concurrency": 4,
            "requests_count": 20
        }
    }

    with open(WORKLOAD_FILE, "w") as f:
        json.dump(workloads, f, indent=2)
    
    print(f">>> Synthetic Data Generated at {WORKLOAD_FILE}")

if __name__ == "__main__":
    generate_configs()