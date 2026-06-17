"""05 · Autonomous Ojin avatar bot — self-converses with NO mic and NO speech-to-text.

Same environment and avatar/LLM/TTS configuration as ``03-pipecat-example``, with two
changes that make it run hands-free for reproducing the avatar's continuous-speech
behaviour (lip-sync / repeated frames):

  1. STT is removed entirely (no Deepgram, no mic input needed).
  2. An ``AutonomousUserDriver`` synthesizes the user side: after every bot turn ends
     it emits a canned utterance as transcription frames, driving the next LLM turn —
     so the bot talks back-to-back, on its own, indefinitely.

    [AutonomousUserDriver] -> LLM -> TTS -> [OjinAvatarService] -> transport.output()

Each avatar session writes a Perfetto trace (see ojin_avatar.py) under
``/root/debug/sessions/stv-example/<date>/<time>_<session_id>/session.json`` — open it
at https://ui.perfetto.dev to inspect the per-tick depth gauges (recv_decode_in/out,
recv_ws_frames, recv_server_msgs, pending_video_frames) across the autonomous turns.

Run (after installing requirements.txt and filling in .env):

    python bot.py              # all transports; open http://localhost:7860/client
    python bot.py -t webrtc    # local WebRTC only — open the page to watch (no mic used)
    python bot.py -t daily     # join a Daily room instead (needs a Daily account)

Open http://localhost:7860/client to watch the avatar; you never need to speak.
"""

import logging
import os
import pathlib

from autonomous_driver import AutonomousUserDriver
from ojin_avatar import OjinAvatarService
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from ojin import MissingCredentialsError, load_env, resolve_credentials

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("ojin-autonomous-bot")

HERE = pathlib.Path(__file__).parent

# The avatar video size. It MUST match the transport's video_out_width/height below
# and your Face model's output size — a mismatch shows up as garbled video.
AVATAR_SIZE = (512, 512)

# System prompt: replies are spoken, so keep them natural and a few sentences long —
# long enough to be a sustained stretch of continuous avatar speech each turn.
SYSTEM_PROMPT = (
    "You are a friendly AI assistant on a live video call. Your replies are spoken "
    "aloud, so answer in two to four natural sentences — long enough to be engaging "
    "but never rambling — and avoid emojis, bullet points, or any formatting that "
    "can't be read out loud."
)

# Load the .env beside this file (Ojin keys + LLM/TTS keys; no STT key needed).
load_env(base_dir=HERE)
try:
    CREDS = resolve_credentials(load_env_file=False)  # OJIN_API_KEY + OJIN_CONFIG_ID
except MissingCredentialsError as exc:
    raise SystemExit(str(exc)) from None


# No audio input is used (no mic / no STT), but the transport still needs to be built;
# we keep audio_in for parity with 03 and so the browser page works unchanged.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_out_enabled=True,
        video_out_is_live=True,
        video_out_width=AVATAR_SIZE[0],
        video_out_height=AVATAR_SIZE[1],
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_out_enabled=True,
        video_out_is_live=True,
        video_out_width=AVATAR_SIZE[0],
        video_out_height=AVATAR_SIZE[1],
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    """Wire the autonomous-driver -> LLM -> TTS -> avatar pipeline and run one call."""
    logger.info("Starting autonomous Ojin avatar bot (no STT)")

    llm = GoogleLLMService(
        api_key=os.environ["GEMINI_API_KEY"],
        model="gemini-2.5-flash",
    )

    tts = ElevenLabsTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        voice_id=os.environ["ELEVENLABS_VOICE_ID"],
        model="eleven_flash_v2_5",
        params=ElevenLabsTTSService.InputParams(
            stability=1.0,
            similarity_boost=1.0,
        ),
    )

    # The Ojin face — same adapter and config as example 03.
    avatar = OjinAvatarService(
        api_key=CREDS.api_key,
        config_id=CREDS.config_id,
        image_size=AVATAR_SIZE,
    )

    # Synthesizes the user side: emits transcription frames after each bot turn.
    driver = AutonomousUserDriver()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = LLMContext(messages)
    # External turn strategies: turn boundaries come from the driver's UserStarted/
    # UserStopped frames, NOT from VAD/STT on live audio (there is none). No VAD
    # analyzer is configured for the same reason.
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=ExternalUserTurnStrategies(),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            driver,  # <-- synthesizes the user turns (replaces STT)
            user_aggregator,
            llm,
            tts,
            avatar,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport: BaseTransport, _client: object) -> None:
        # No greeting is queued here: the AutonomousUserDriver is the single source of
        # turns. It bootstraps the opening turn a few seconds after StartFrame (so the
        # avatar has connected), then chains a new turn after each BotStoppedSpeaking —
        # which also means the loop runs headless, with or without a browser watching.
        logger.info("Client connected — the bot is self-driving the conversation")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(
        _transport: BaseTransport, _client: object
    ) -> None:
        logger.info("Client disconnected — ending the call")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments) -> None:
    """Runner entry point: build the chosen transport, then run the pipeline."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
