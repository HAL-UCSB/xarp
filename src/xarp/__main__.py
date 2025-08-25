import sys
import pathlib
from streamlit.web import cli


def main():
    script = pathlib.Path(__file__).with_name('web_client') / 'app.py'
    sys.argv = ['streamlit', 'run', str(script)]
    sys.exit(cli.main())


if __name__ == '__main__':
    main()
