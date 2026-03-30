"""
demo.py - Easy demo script for presentation
Run: python demo.py
"""
import requests
import uuid

BASE_URL = "http://localhost:8000"
AGENT_B_URL = "http://localhost:8001"


def ask(message: str, session_id: str = None) -> None:
    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    print(f"\n{'='*60}")
    print(f"QUESTION: {message}")
    print(f"{'='*60}")

    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"message": message, "session_id": session_id},
            timeout=120
        )
        data = response.json()
        print(f"ROUTE: {data.get('route', 'unknown')}")
        print(f"\nCOACH: {data.get('response', 'No response')}")
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out. Ollama may be slow.")
    except Exception as e:
        print(f"ERROR: {str(e)}")


def check_progress() -> None:
    print(f"\n{'='*60}")
    print("CHECKING PROGRESS: Bench Press over 3 weeks")
    print(f"{'='*60}")

    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            json={
                "message": "Am I making progress?",
                "session_id": "demo_progress",
                "progress_goal": "strength",
                "progress_logs": [
                    {"date": "2026-03-01", "exercise": "Bench Press", "weight": 40, "reps": 8, "sets": 3},
                    {"date": "2026-03-08", "exercise": "Bench Press", "weight": 45, "reps": 8, "sets": 3},
                    {"date": "2026-03-15", "exercise": "Bench Press", "weight": 50, "reps": 8, "sets": 3},
                ]
            },
            timeout=120
        )
        data = response.json()
        print(f"ROUTE: {data.get('route', 'unknown')}")
        print(f"\nANALYSIS: {data.get('response', 'No response')}")
    except Exception as e:
        print(f"ERROR: {str(e)}")


def check_plateau() -> None:
    print(f"\n{'='*60}")
    print("CHECKING PLATEAU: Squat same weight 4 sessions")
    print(f"{'='*60}")

    try:
        response = requests.post(
            f"{AGENT_B_URL}/analyze-progress",
            json={
                "session_id": "demo_plateau",
                "progress_goal": "strength",
                "progress_logs": [
                    {"date": "2026-03-01", "exercise": "Squat", "weight": 80, "reps": 5, "sets": 3},
                    {"date": "2026-03-08", "exercise": "Squat", "weight": 80, "reps": 5, "sets": 3},
                    {"date": "2026-03-15", "exercise": "Squat", "weight": 80, "reps": 5, "sets": 3},
                    {"date": "2026-03-22", "exercise": "Squat", "weight": 80, "reps": 5, "sets": 3},
                ]
            },
            timeout=30
        )
        data = response.json()
        print(f"SUMMARY: {data.get('summary', 'No response')}")
        print(f"PLATEAUS: {data.get('plateaus', [])}")
        print(f"RECOMMENDATIONS: {data.get('recommendations', '')}")
    except Exception as e:
        print(f"ERROR: {str(e)}")


def run_full_demo():
    print("\n FITNESS COACH AI - LIVE DEMO")
    print("="*60)

    print("\n DEMO 1: Exercise Agent (RAG + LLM)")
    ask("What are the best chest exercises I can do with a barbell?")
    input("\nPress Enter for next demo...")

    print("\n DEMO 2: Nutrition Agent (RAG + LLM)")
    ask("How much protein should I eat to build muscle?")
    input("\nPress Enter for next demo...")

    print("\n DEMO 3: Program Builder Agent (LLM Synthesis)")
    ask("Build me a 3 day beginner workout program")
    input("\nPress Enter for next demo...")

    print("\n DEMO 4: Agent A calling Agent B via HTTP")
    check_progress()
    input("\nPress Enter for next demo...")

    print("\n DEMO 5: Plateau Detection (Agent B directly)")
    check_plateau()

    print(f"\n{'='*60}")
    print("DEMO COMPLETE")
    print("="*60)


def interactive():
    print("\n FITNESS COACH AI - INTERACTIVE CHAT")
    print("="*60)
    print("Type any fitness question and get a response.")
    print("Type 'progress' to run a progress analysis demo.")
    print("Type 'plateau' to run a plateau detection demo.")
    print("Type 'quit' to exit.")
    print("="*60)

    session_id = str(uuid.uuid4())[:8]
    print(f"\nSession ID: {session_id}\n")

    while True:
        try:
            message = input("YOU: ").strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            break

        if not message:
            continue
        if message.lower() in ('quit', 'thanks', 'bye', 'goodbye', 'gd bye', 'good bye'):
                print("Goodbye!")
                break
        elif message.lower() == 'progress':
            check_progress()
        elif message.lower() == 'plateau':
            check_plateau()
        else:
            ask(message, session_id=session_id)


if __name__ == "__main__":
    print("\n FITNESS COACH AI - DEMO LAUNCHER")
    print("="*60)
    print("1. Run full automated demo (5 pre-built queries)")
    print("2. Interactive chat (type your own questions)")
    print("="*60)

    choice = input("\nChoose (1 or 2): ").strip()

    if choice == "2":
        interactive()
    elif choice == "1":
        run_full_demo()
    else:
        print("Invalid choice. Running full demo by default...")
        run_full_demo()