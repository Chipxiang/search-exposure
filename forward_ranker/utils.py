import datetime
from datetime import datetime, timezone, timedelta

TIME_OFFSET = -4


def print_message(s, offset=TIME_OFFSET):
    print("[{}] {}".format(datetime.now(timezone(timedelta(hours=offset))).strftime("%b %d, %H:%M:%S"), s), flush=True)