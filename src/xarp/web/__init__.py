import sys
from streamlit.web import cli as stcli


def run():
    sys.argv = ['streamlit', 'run', str('app.py')]
    sys.exit(stcli.main())
