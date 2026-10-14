"""Start Hydra.ai:  python run.py   then open http://127.0.0.1:8000"""
import os
import threading
import webbrowser

import uvicorn


def main():
    host, port = "127.0.0.1", int(os.getenv("PORT", "8000"))
    threading.Timer(2.0, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run("hydra.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
