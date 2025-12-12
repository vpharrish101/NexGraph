import asyncio
import json
import websockets
from utils import cfg_logging

WS_URL="ws://127.0.0.1:8000/ws/run/<uuid>"

async def run_test():
    text='''To embark upon the construction of a single, continuous sentence that measures exactly three hundred words is to accept a challenge that tests the very limits of syntactic endurance, requiring the writer to weave together a sprawling tapestry of independent and dependent clauses that stretch across the page like a linguistic horizon, utilizing every available conjunction and punctuation mark—save for the final period—to maintain a cohesive flow that prevents the structure from collapsing into a chaotic jumble of nonsense, a task that demands a certain degree of stubbornness as one navigates through the dense forest of vocabulary, selecting the precise adjectives and adverbs to elaborate on the mundane details of the writing process itself, such as the rhythmic clatter of the keyboard which mimics the heartbeat of this growing textual organism, or the way the cursor blinks with an impatient regularity, urging the creator onward through digressions about the nature of infinity and the arbitrary constraints we place upon art, all while balancing the cognitive load of holding multiple threads of thought in suspension, ensuring that the reader, who has graciously volunteered their attention, does not become hopelessly lost in the maze of verbiage but is instead guided gently, if somewhat relentlessly, toward the distant conclusion, a destination that seems to recede with every added phrase, forcing a reliance on the semicolon to act as a temporary resting place, a ledge on a cliff face where one might catch their breath before ascending the next peak of exposition, demonstrating that language is a flexible tool capable of infinite expansion, provided one possesses the patience to stitch the pieces together until the final target is visible, allowing the weary syntax to resolve itself and, with a sense of profound relief, finally come to a complete and total stop right now.'''
    payload={
        "state":{
            "text":text,
            "lim":100
        }
    }

    try:
        async with websockets.connect(WS_URL,ping_interval=20,ping_timeout=20) as ws:
            await ws.send(json.dumps(payload))
            try:
                while True:
                    msg=await asyncio.wait_for(ws.recv(),timeout=60)
                    if isinstance(msg, bytes):
                        try:
                            msg=msg.decode("utf-8")
                        except Exception:
                            print("<binary message received>")
                            continue
                    
            except asyncio.TimeoutError:
                print("\nNo messages for 60s")
                cfg_logging.logger.error("<ws> Timeout, no msg for 60s :(")
            except websockets.ConnectionClosedOK:
                print("\nServer closed connection (normal).")
                cfg_logging.logger.info("<ws> Closed normally :)")
            except websockets.ConnectionClosedError as e:
                print("\nServer closed connection with error:", e)
                cfg_logging.logger.error(f"<ws> Closed with error: {e} :(")

    except Exception as e:
        print("Connection failed:", e)
        cfg_logging.logger.error(f"<ws> Connection failed with error: {e} :(")


if __name__ == "__main__":
    asyncio.run(run_test())
