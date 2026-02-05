import chainlit as cl
import httpx

BACKEND_URL = "http://backend:8001"

@cl.on_message
async def on_message(message: cl.Message):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{BACKEND_URL}/rag",
            json={"question": message.content},
            timeout=60,
        )
    answer = resp.json()["answer"]
    await cl.Message(content=answer).send()
