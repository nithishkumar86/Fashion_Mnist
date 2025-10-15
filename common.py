import subprocess
import threading
import time
import sys

def run_backround():
    try:
        subprocess.run(["uvicorn","Backend.backend_server:app","--host","127.0.0.1","--port","8000"],check=True)
    except Exception as e:  # check = True => makes Python throw an exception if the command fails.
        print(f"an error occured {str(e)}")

def run_frontend():
    try:
        subprocess.run(["streamlit","run","streamlits.py"],check=True)
    except Exception as e:
        print(f"an error occured {str(e)}")


if __name__ == "__main__":
    threading.Thread(target=run_backround).start()
    time.sleep(2)
    run_frontend()




