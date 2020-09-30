import pickle
import datetime
from datetime import datetime, timezone, timedelta


TIME_OFFSET = -4

def print_message(s, offset=TIME_OFFSET):
    print("[{}] {}".format(datetime.now(timezone(timedelta(hours=offset))).strftime("%b %d, %H:%M:%S"), s), flush=True)

def obj_reader(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle, encoding="bytes")

def obj_writer(obj, path):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)