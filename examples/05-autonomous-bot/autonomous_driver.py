"""Autonomous conversation driver: keeps the bot talking with no human, no STT.

This example reproduces the avatar's continuous-speech behaviour (lip-sync / repeated
frames) without a person at the mic. Instead of speech-to-text, a small FrameProcessor
*synthesizes* the user side: after every bot turn ends (``BotStoppedSpeakingFrame``) it
emits a canned user utterance as real transcription frames, which drives the next LLM
turn. The result is a self-sustaining back-to-back conversation that exercises the
``OjinAvatarService`` -> ``OjinSTVClient`` pipeline turn after turn.

Wire it FIRST in the pipeline (right after ``transport.input()``), and configure the
user aggregator with ``ExternalUserTurnStrategies()`` so turn boundaries come from the
``UserStartedSpeakingFrame`` / ``UserStoppedSpeakingFrame`` this driver emits rather than
from VAD/STT on live audio (there is none):

    transport.input() -> [AutonomousUserDriver] -> user_agg -> LLM -> TTS -> avatar -> out

Each synthetic turn is emitted DOWNSTREAM as the exact frame sequence real STT + VAD
would produce, so nothing downstream knows the difference:

    UserStartedSpeakingFrame
    TranscriptionFrame(text=<canned>, finalized=True)
    UserStoppedSpeakingFrame   # ExternalUserTurnStopStrategy finalizes -> LLM runs
"""

from __future__ import annotations

import asyncio
import logging

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.time import time_now_iso8601

logger = logging.getLogger(__name__)

# Canned user turns, rotated round-robin. They ask for short explanations so each bot
# reply is a few seconds of continuous speech — enough sustained frames to surface the
# repeat/lip-sync behaviour, turn after turn, while staying varied so the LLM does not
# fall into a loop.
DEFAULT_UTTERANCES = [
    "Tell me an interesting fact about the ocean.",
    "What's a simple recipe I could make for dinner tonight?",
    "Explain in a couple of sentences how rainbows form.",
    "Recommend a book and say why I'd like it.",
    "What's a good way to stay focused while working?",
    "Describe your favorite season and what makes it special.",
    "Give me a fun piece of trivia about space.",
    "Suggest a beginner-friendly hobby and how to start it.",
    "What's one small habit that can improve someone's day?",
    "Tell me a short, light-hearted joke.",
]


class AutonomousUserDriver(FrameProcessor):
    """Synthesize the user side of the conversation so the bot self-converses.

    Place first in the pipeline. Bootstraps the first turn shortly after start, then
    fires the next synthetic user turn after each ``BotStoppedSpeakingFrame``.
    """

    def __init__(
        self,
        *,
        utterances: list[str] | None = None,
        initial_delay_s: float = 3.0,
        inter_turn_delay_s: float = 0.6,
        max_turns: int | None = None,
    ) -> None:
        """Create the driver.

        Args:
            utterances: Canned user turns, rotated round-robin. Defaults to
                :data:`DEFAULT_UTTERANCES`.
            initial_delay_s: Delay after ``StartFrame`` before the first synthetic
                turn (lets the transport + avatar connect first).
            inter_turn_delay_s: Pause after the bot stops speaking before the next
                synthetic turn, so each turn has a clean gap.
            max_turns: Stop driving after this many synthetic turns (``None`` = run
                forever, until the call ends).

        """
        super().__init__(name="autonomous-user-driver")
        self._utterances = utterances or DEFAULT_UTTERANCES
        self._initial_delay_s = initial_delay_s
        self._inter_turn_delay_s = inter_turn_delay_s
        self._max_turns = max_turns
        self._turn_index = 0
        self._started = False
        self._pending: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Pass frames through; bootstrap on Start, drive on each bot-stopped."""
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            # Push Start through first, then arm the opening turn.
            await self.push_frame(frame, direction)
            self._started = True
            self._schedule_turn(self._initial_delay_s)
            return

        if isinstance(frame, (EndFrame, CancelFrame)):
            self._cancel_pending()
            await self.push_frame(frame, direction)
            return

        # Each completed bot turn arms the next synthetic user turn. BotStopped is
        # pushed both up- and downstream by the output transport; react once and only
        # when nothing is already queued (so a double-fire can't stack two turns).
        if isinstance(frame, BotStoppedSpeakingFrame):
            await self.push_frame(frame, direction)
            if self._started and (self._pending is None or self._pending.done()):
                self._schedule_turn(self._inter_turn_delay_s)
            return

        await self.push_frame(frame, direction)

    def _schedule_turn(self, delay_s: float) -> None:
        """Arm the next synthetic user turn after ``delay_s`` seconds."""
        if self._max_turns is not None and self._turn_index >= self._max_turns:
            logger.info("Autonomous driver reached max_turns=%s; stopping", self._max_turns)
            return
        self._cancel_pending()
        self._pending = self.create_task(self._emit_turn_after(delay_s))

    def _cancel_pending(self) -> None:
        """Cancel any armed-but-not-yet-emitted turn."""
        if self._pending is not None and not self._pending.done():
            self.cancel_task(self._pending)
        self._pending = None

    async def _emit_turn_after(self, delay_s: float) -> None:
        """Wait, then emit one synthetic user turn as transcription frames."""
        await asyncio.sleep(delay_s)
        text = self._utterances[self._turn_index % len(self._utterances)]
        self._turn_index += 1
        logger.info("Autonomous user turn %d: %r", self._turn_index, text)

        # The exact sequence real STT + VAD produce. ExternalUserTurnStrategies on the
        # user aggregator turns UserStarted/UserStopped into the turn boundaries and
        # runs the LLM on stop.
        await self.push_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await self.push_frame(
            TranscriptionFrame(
                text=text,
                user_id="autonomous",
                timestamp=time_now_iso8601(),
                finalized=True,
            ),
            FrameDirection.DOWNSTREAM,
        )
        await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
