import time
import random
import hashlib

class General:
    @staticmethod
    def guid(*args):
        a = str(time.time())
        b = str(random.random())
        result = a + b + str(args)
        return hashlib.md5(result.encode('utf-8')).hexdigest()
