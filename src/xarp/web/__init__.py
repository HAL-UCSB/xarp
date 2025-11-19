from pathlib import Path
import sys
from streamlit.web import cli as stcli


def run():
    script = Path(__file__).parent / 'responses.py'
    print(script)
    sys.argv = ['streamlit', 'run', str(script)]
    sys.exit(stcli.main())
