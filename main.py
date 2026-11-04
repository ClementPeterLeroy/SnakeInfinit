"""Run with: python main.py"""
import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def main():
    # Spawn imports this entry point as __mp_main__; only the UI parent loads Pygame.
    from app import main as application_main
    application_main()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
