"""Local proxy connection integration test."""

import asyncio
import logging
import os
import sys

import numpy as np

from ojin.ojin_client import OjinClient
from ojin.ojin_client_messages import (
    OjinAudioInputMessage,
    OjinEndInteractionMessage,
    OjinInteractionResponseMessage,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ORIS_SERVER_URL: str = "ws://0.0.0.0:8000/realtime"


async def main() -> bool:
    """Connect to a local proxy and run a short audio interaction."""
    api_key = os.environ.get("CONNECT_API_KEY")
    config_id = os.environ.get(
        "CONNECT_CONFIG_ID",
        "7d331075-cacb-41e7-8e2d-4c9dd9eb9090",
    )

    if not api_key:
        logger.error("CONNECT_API_KEY environment variable is not set")
        msg = "CONNECT_API_KEY environment variable must be set"
        raise ValueError(msg)

    client = OjinClient(
        ws_url=ORIS_SERVER_URL,
        api_key=api_key,
        config_id=config_id,
        reconnect_attempts=1,
        reconnect_delay=1.0,
    )

    await client.connect()

    connect_message = await client.receive_message()
    logger.info("connect Mes: %s", connect_message)

    await client.start_interaction()

    duration = 1.0  # seconds
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    frequency = 440.0  # A4 note
    test_audio_samples = (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

    chunk_size = 3200

    num_chunks = len(test_audio_samples) // chunk_size
    logger.info("%d", num_chunks)

    try:
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk = test_audio_samples[start_idx:end_idx]

            logger.info("%d", len(chunk.tobytes()))

            # Convert float32 samples to int16 (scale by 32767)
            chunk_int16 = (chunk * 32767).astype(np.int16)
            interaction_messsage = OjinAudioInputMessage(
                audio_int16_bytes=chunk_int16.tobytes(),
                params=None,
            )

            await client.send_message(interaction_messsage)

        await client.send_message(OjinEndInteractionMessage())

        final_response = False

        try:
            while not final_response:
                first_mes = await asyncio.wait_for(
                    client.receive_message(), timeout=1.0
                )
                assert isinstance(first_mes, OjinInteractionResponseMessage)
                logger.info(
                    "interaction_msg : %s",
                    first_mes.is_final_response,
                )
                final_response = first_mes.is_final_response

            await client.close()
            return True

        except asyncio.TimeoutError as err:
            raise ConnectionError("ConnectionLost") from err

    except ConnectionError:
        raise
    except Exception as err:
        raise RuntimeError("error") from err


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Test runner crashed: %s", e)
        sys.exit(1)
