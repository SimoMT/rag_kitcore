from chainlit import on_message, Message
import httpx

BACKEND_URL = "http://backend:8001"

client = httpx.AsyncClient(timeout=180)


@on_message
async def handle_message(message: Message):
    try:
        response = await client.post(
            f"{BACKEND_URL}/rag",
            json={"question": message.content},
        )

        response.raise_for_status()
        data = response.json()
        answer = data.get("answer", "No answer returned by backend.")

        await Message(content=answer).send()

    except httpx.HTTPError as e:
        await Message(
            content=f"Backend error: {str(e)}"
        ).send()
