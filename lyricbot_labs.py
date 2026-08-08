import sys
from channel_config import build_config
import lyricbot_core

def main():
    return lyricbot_core.run_workflow_and_exit(build_config('labs'))

if __name__ == '__main__':
    sys.exit(main())
