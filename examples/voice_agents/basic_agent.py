import logging
from dotenv import load_dotenv
import os

print("CWD:", os.getcwd())
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))
print("LIVEKIT_URL:", os.getenv("LIVEKIT_URL"))

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    cli,
    metrics,
    room_io,
)

from livekit.agents.llm import function_tool
from livekit.plugins import silero


# --------------------------------------------------
# GLOBAL STATE + CONFIG
# --------------------------------------------------

# Tracks whether the agent is currently speaking
agent_is_speaking = False

# Passive acknowledgement words (to ignore while speaking)
IGNORE_WORDS = [
    "yeah", "ok", "okay", "hmm", "uh-huh", "right"
]

# Real interruption commands
INTERRUPT_WORDS = [
    "stop", "wait", "no", "hold on"
]

logger = logging.getLogger("basic-agent")
logger.setLevel(logging.INFO)

load_dotenv()


# --------------------------------------------------
# AGENT DEFINITION
# --------------------------------------------------

class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Kelly. You interact with users via voice. "
                "Keep responses concise and to the point. "
                "Do not use emojis, asterisks, markdown, or special characters. "
                "You are curious, friendly, and have a sense of humor. "
                "You speak English."
            ),
        )

    async def on_enter(self):
        """
        Called when the agent enters the session.
        Agent starts speaking here.
        """
        global agent_is_speaking
        agent_is_speaking = True
        logger.info("Agent started speaking")
        self.session.generate_reply()

    async def on_response_complete(self):
        """
        Called when the agent finishes speaking.
        """
        global agent_is_speaking
        agent_is_speaking = False
        logger.info("Agent finished speaking")

    async def on_message(self, message: str):
        """
        Intercepts user transcription text.
        This is where we implement ignore vs interrupt logic.
        """
        global agent_is_speaking

        text = message.lower().strip()
        words = text.split()

        contains_interrupt = any(word in text for word in INTERRUPT_WORDS)
        only_soft_words = len(words) > 0 and all(word in IGNORE_WORDS for word in words)

        logger.info(
            f"User said: '{text}' | speaking={agent_is_speaking} | "
            f"interrupt={contains_interrupt} | soft_only={only_soft_words}"
        )

        # CASE 1: Agent is currently speaking
        if agent_is_speaking:
            if contains_interrupt:
                logger.info("→ REAL INTERRUPTION (agent will stop)")
                return message            # allow interruption
            elif only_soft_words:
                logger.info("→ PASSIVE ACKNOWLEDGEMENT (ignored)")
                return None               # completely ignore
            else:
                logger.info("→ SEMANTIC INTERRUPTION")
                return message            # interrupt normally

        # CASE 2: Agent is silent
        logger.info("→ Agent silent, processing normally")
        return message

    # Example tool (unchanged)
    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ):
        logger.info(f"Looking up weather for {location}")
        return "sunny with a temperature of 70 degrees."


# --------------------------------------------------
# SERVER SETUP
# --------------------------------------------------

server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt="deepgram/nova-3",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,

        # IMPORTANT: helps resume speech after VAD false triggers
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
    )

    # Metrics logging (unchanged)
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
