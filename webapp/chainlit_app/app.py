import chainlit as cl
from chainlit_app.rag_pipeline import rag

@cl.on_message
async def main(message):
    answer = rag.query(message.content)
    await cl.Message(content=answer).send()
