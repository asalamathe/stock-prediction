import sys
from streamlit.web.cli import main
from pathlib import Path

file_path = Path(__file__).parent / "app.py"

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "--server.headless", "true", str(file_path)]
    main()