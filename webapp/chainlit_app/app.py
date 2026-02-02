# --------------------------------------------------
# Temporary Fix on relative imports issue
# --------------------------------------------------

import sys
from pathlib import Path

# Resolve the project root (two levels up from this file)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -------------------------------------------------
# 

import chainlit as cl
from webapp.chainlit_app.rag_pipeline import rag


@cl.on_message
async def main(message):
    answer = rag.run(message.content)
    await cl.Message(content=answer).send()

# import chainlit as cl
# from webapp.chainlit_app.rag_pipeline import rag

# @cl.on_message
# async def main(message):
#     async with cl.Message(content="") as msg:
#         for chunk in rag.stream(message.content):
#             await msg.stream_token(chunk)
