#!/bin/sh

ollama serve &
SERVER_PID=$!

sleep 3

ollama pull llama3.2:1b

wait $SERVER_PID
