"""Test that both WebSocket server and client start up correctly."""

import asyncio
import logging
import multiprocessing
import sys

from pydantic import BaseModel

from ojin.entities.interaction_messages import (
    ErrorResponseMessage,
)
from ojin.ojin_client import (
    OjinClient,
)
from ojin.ojin_client_messages import (
    OjinAudioInputMessage,
    OjinInteractionResponseMessage,
    OjinSessionReadyMessage,
)
from tests.mock.inference_proxy_mock import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class UnknownMessage(BaseModel):
    """Unknown message type used for error-path testing."""

    type: str = "Unknown"


def run_server():
    """Run the FastAPI WebSocket server in a separate process."""
    try:
        import uvicorn

        # Try to import your server module
        logger.info("Starting FastAPI WebSocket server on port 8000...")
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    except Exception as e:
        logger.error("Error starting server: %s", e)
        sys.exit(1)


def create_client():
    """Run the existing OJIN Avatar client in a separate process."""
    logger.info("Initializing OJIN Avatar WebSocket client...")

    # Initialize client with dummy parameters
    client = OjinClient(
        ws_url="ws://127.0.0.1:8000/ws",
        api_key="test-api-key",
        config_id="test-config-id",
        reconnect_attempts=1,
        reconnect_delay=1.0,
    )

    logger.info("Client initialized successfully")
    return client


def _terminate(process, name):
    """Terminate a multiprocessing.Process if it's still alive."""
    if process and process.is_alive():
        logger.info("Terminating %s process...", name)
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            logger.warning("Force killing %s process...", name)
            process.kill()
            process.join()


async def start_server_n_client():
    """Start the server and the client on separate processes.

    On success, returns a 5-tuple: (client, task_queue, done_queue,
    client_process, server_process). On failure, returns None and ensures
    any spawned processes are torn down before returning.
    """
    server_process = None
    client_process = None
    try:
        logger.info("Starting server process...")
        server_process = multiprocessing.Process(
            target=run_server, name="WebSocketServer"
        )

        server_process.start()
        await asyncio.sleep(1)

        if not server_process.is_alive():
            logger.error("Server failed to start")
            _terminate(server_process, "server")
            return None

        logger.info("Server is running")

        # Start client
        task_queue = multiprocessing.Queue()
        done_queue = multiprocessing.Queue()

        logger.info("Starting client process...")
        client = create_client()

        async def connect(input_queue, done_queue):
            await client.connect()
            await asyncio.sleep(1)

            msg = await client.receive_message()
            done_queue.put(msg)

            for task in iter(input_queue.get, "STOP"):
                try:
                    if task == "GET_TOKEN":
                        await client.start_interaction()
                        continue

                    await client.send_message(task)
                    msg = await client.receive_message()
                except Exception as e:
                    done_queue.put(e)
                    break

                done_queue.put(msg)

            await asyncio.sleep(1)

        def start(input_queue, done_queue):
            asyncio.run(connect(input_queue, done_queue))

        client_process = multiprocessing.Process(
            target=start,
            name="WebSocketClient",
            args=(task_queue, done_queue),
        )
        client_process.start()
        await asyncio.sleep(2)

        if not client_process.is_alive():
            logger.error("Client failed to start")
            _terminate(client_process, "client")
            _terminate(server_process, "server")
            return None

        logger.info("Client is running")
        logger.info("Both server and client are running simultaneously")

        return (
            client,
            task_queue,
            done_queue,
            client_process,
            server_process,
        )

    except Exception as e:
        logger.error("Could not start the processes: %s", e)
        _terminate(client_process, "client")
        _terminate(server_process, "server")
        return None


async def test_client_connection():
    """Test that the client can connect to the inference server."""
    logger.info("=" * 60)
    logger.info("TEST 1: Server and Client Running Simultaneously")
    logger.info("=" * 60)

    client_process = None
    server_process = None
    try:
        result = await start_server_n_client()
        if not result:
            return False
        (
            _client,
            _client_task_queue,
            client_done_queue,
            client_process,
            server_process,
        ) = result

        client_msg = client_done_queue.get()
        if not isinstance(client_msg, OjinSessionReadyMessage):
            return False

        await asyncio.sleep(1)

        return True
    finally:
        _terminate(client_process, "client")
        _terminate(server_process, "server")
        logger.info("All processes cleaned up")


async def test_client_error_message():
    """Test that the client handles error messages from the server."""
    logger.info("=" * 60)
    logger.info("TEST 2: Server and Client Running Simultaneously")
    logger.info("=" * 60)

    client_process = None
    server_process = None
    try:
        result = await start_server_n_client()
        if not result:
            return False
        (
            _client,
            client_task_queue,
            client_done_queue,
            client_process,
            server_process,
        ) = result

        client_msg = client_done_queue.get()
        if not isinstance(client_msg, OjinSessionReadyMessage):
            return False

        interaction_messsage = UnknownMessage()

        client_task_queue.put(interaction_messsage)
        client_task_queue.put("STOP")
        client_msg = client_done_queue.get()

        if not (
            isinstance(client_msg, Exception)
            and isinstance(client_msg.args[0], ErrorResponseMessage)
        ):
            return False

        await asyncio.sleep(1)

        return True
    except Exception as e:
        logger.error("Test crashed: %s", e)
        return False
    finally:
        _terminate(client_process, "client")
        _terminate(server_process, "server")
        logger.info("All processes cleaned up")


async def test_client_interaction():
    """Test that the client can send interactions to the server."""
    logger.info("=" * 60)
    logger.info("TEST 3: Client sends interaction message to the server")
    logger.info("=" * 60)

    client_process = None
    server_process = None
    try:
        result = await start_server_n_client()
        if not result:
            return False
        (
            _client,
            client_task_queue,
            client_done_queue,
            client_process,
            server_process,
        ) = result

        client_msg = client_done_queue.get()
        if not isinstance(client_msg, OjinSessionReadyMessage):
            return False

        client_task_queue.put("GET_TOKEN")

        test_audio_bytes = b"\x00\x01\x02\x03\x04\x05" * 100
        interaction_messsage = OjinAudioInputMessage(
            audio_int16_bytes=test_audio_bytes,
            params=None,
        )

        client_task_queue.put(interaction_messsage)
        client_task_queue.put("STOP")
        client_msg = client_done_queue.get()

        if not isinstance(client_msg, OjinInteractionResponseMessage):
            return False

        await asyncio.sleep(1)

        return True
    except Exception as e:
        logger.error("Test crashed: %s", e)
        return False
    finally:
        _terminate(client_process, "client")
        _terminate(server_process, "server")
        logger.info("All processes cleaned up")


async def main():
    """Run all startup tests."""
    logger.info("Starting WebSocket Server and Client Startup Tests")
    logger.info("=" * 60)

    results = []

    # Test 1: Connects to the inference proxy mock
    try:
        result = await test_client_connection()
        results.append(("test_client_connection", result))
    except Exception as e:
        logger.error("Test crashed: %s", e)
        results.append(("test_client_connection", False))

    # Test 2: Connects to the inference proxy mock
    try:
        result = await test_client_error_message()
        results.append(("test_client_error_message", result))
    except Exception as e:
        logger.error("Test crashed: %s", e)
        results.append(("test_client_error_message", False))

    # Test 3: Sends interaction messages to the inference proxy
    try:
        result = await test_client_interaction()
        results.append(("test_client_interaction", result))
    except Exception as e:
        logger.error("Test crashed: %s", e)
        results.append(("test_client_interaction", False))

    # Print summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info("%-20s : %s", test_name, status)
        if not result:
            all_passed = False

    logger.info("=" * 60)
    if all_passed:
        logger.info("ALL TESTS PASSED! Both server and client can start successfully.")
    else:
        logger.warning("Some tests failed. Check the logs above for details.")

    return all_passed


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
