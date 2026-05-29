from __future__ import annotations

import socket

import uvicorn


HOST = "127.0.0.1"
START_PORT = 8000


def find_available_port(start_port: int) -> int:
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex((HOST, port)) != 0:
                return port
        port += 1


if __name__ == "__main__":
    port = find_available_port(START_PORT)
    print(f"Indian Stock Price Finder running at http://{HOST}:{port}")
    uvicorn.run("app.main:app", host=HOST, port=port, reload=True)
