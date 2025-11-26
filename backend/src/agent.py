# ======================================================
# 💼 DAY 5: AI SALES DEVELOPMENT REP (SDR)
# 👨‍💻 "Arvind Store" - Auto-Lead Capture Agent
# 🚀 Features: FAQ Retrieval, Lead Qualification, JSON Database
# ======================================================

import logging
import json
import os
import asyncio
from datetime import datetime
from typing import Annotated, Optional
from dataclasses import dataclass, asdict

print("\n" + "💼" * 50)
print("🚀 AI SDR AGENT - DAY 5 TUTORIAL")
print("📚 SELLING: Arvind's AI, Chatbot & Web Development Courses")
print("💡 agent.py LOADED SUCCESSFULLY!")
print("💼" * 50 + "\n")

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

# 🔌 PLUGINS
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ======================================================
# 📂 1. KNOWLEDGE BASE (FAQ)
# ======================================================

FAQ_FILE = "arvind_store_faq.json"
LEADS_FILE = "arvind_leads_db.json"

DEFAULT_FAQ = [
    {
        "question": "What do you sell?",
        "answer": "I offer professional training in AI, Chatbot Development, Java Web Development and Voice AI Agents. I also provide project guidance and mentorship for students."
    },
    {
        "question": "How much does the AI course cost?",
        "answer": "The AI & Chatbot Development course starts from ₹4,999 depending on the module depth and duration."
    },
    {
        "question": "Do you provide certificates?",
        "answer": "Yes, all paid courses include a verified completion certificate and project mentoring support."
    },
    {
        "question": "Do you offer project help?",
        "answer": "Yes, I help students build real-world AI and chatbot projects for academic and professional use."
    }
]

def load_knowledge_base():
    try:
        path = os.path.join(os.path.dirname(__file__), FAQ_FILE)
        if not os.path.exists(path):
            with open(path, "w", encoding='utf-8') as f:
                json.dump(DEFAULT_FAQ, f, indent=4)
        with open(path, "r", encoding='utf-8') as f:
            return json.dumps(json.load(f))
    except Exception as e:
        print(f"⚠️ Error loading FAQ: {e}")
        return ""

STORE_FAQ_TEXT = load_knowledge_base()

# ======================================================
# 💾 2. LEAD DATA STRUCTURE
# ======================================================

@dataclass
class LeadProfile:
    name: str | None = None
    company: str | None = None
    email: str | None = None
    role: str | None = None
    use_case: str | None = None
    team_size: str | None = None
    timeline: str | None = None
   
    def is_qualified(self):
        return all([self.name, self.email, self.use_case])

@dataclass
class Userdata:
    lead_profile: LeadProfile

# ======================================================
# 🛠️ 3. SDR TOOLS
# ======================================================

@function_tool
async def update_lead_profile(
    ctx: RunContext[Userdata],
    name: Annotated[Optional[str], Field(description="Customer's name")] = None,
    company: Annotated[Optional[str], Field(description="Customer's company name")] = None,
    email: Annotated[Optional[str], Field(description="Customer's email address")] = None,
    role: Annotated[Optional[str], Field(description="Customer's job title")] = None,
    use_case: Annotated[Optional[str], Field(description="What they want to build or learn")] = None,
    team_size: Annotated[Optional[str], Field(description="Number of people in their team")] = None,
    timeline: Annotated[Optional[str], Field(description="When they want to start")] = None,
) -> str:
    profile = ctx.userdata.lead_profile

    if name: profile.name = name
    if company: profile.company = company
    if email: profile.email = email
    if role: profile.role = role
    if use_case: profile.use_case = use_case
    if team_size: profile.team_size = team_size
    if timeline: profile.timeline = timeline

    print(f"📝 UPDATING LEAD: {profile}")
    return "Lead profile updated successfully."

@function_tool
async def submit_lead_and_end(ctx: RunContext[Userdata]) -> str:
    profile = ctx.userdata.lead_profile

    db_path = os.path.join(os.path.dirname(__file__), LEADS_FILE)
    entry = asdict(profile)
    entry["timestamp"] = datetime.now().isoformat()

    existing_data = []
    if os.path.exists(db_path):
        try:
            with open(db_path, "r") as f:
                existing_data = json.load(f)
        except: pass

    existing_data.append(entry)

    with open(db_path, "w") as f:
        json.dump(existing_data, f, indent=4)

    print(f"✅ LEAD SAVED TO {LEADS_FILE}")
    return f"Thanks {profile.name}! I have recorded your interest in {profile.use_case}. We will contact you soon at {profile.email}. Goodbye!"

# ======================================================
# 🧠 4. AGENT DEFINITION
# ======================================================

class SDRAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=f"""
            You are 'Arvind', a friendly Sales Development Representative for 'Arvind Store'.

            📘 KNOWLEDGE BASE:
            {STORE_FAQ_TEXT}

            🎯 GOAL:
            - Answer questions about AI, Chatbot & Web Development courses.
            - Collect lead details naturally.

            Ask for:
            - Name
            - Email
            - Role / Company
            - Use Case
            - Timeline

            ⚙️ Behavior:
            - Be polite, natural and helpful.
            - After answering, ask one lead question.
            - Use update_lead_profile tool when user provides info.
            - When user says bye/thanks/done, call submit_lead_and_end.

            🚫 Never invent answers not in FAQ.
            """,
            tools=[update_lead_profile, submit_lead_and_end],
        )

# ======================================================
# 🎬 ENTRYPOINT
# ======================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    print("\n🚀 STARTING ARVIND SDR SESSION")

    userdata = Userdata(lead_profile=LeadProfile())

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-IN-aarav",
            style="Conversational",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )

    await session.start(
        agent=SDRAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
