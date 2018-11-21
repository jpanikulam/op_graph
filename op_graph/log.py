try:
    from colorama import init as colorama_init
    from colorama import Fore, Style
    colorama_init()

except(ImportError):
    print "Could not import colorama: Skipping coloration"

    class Dummy(object):
        def __getattr__(self, key):
            return ''

    Fore = Dummy()
    Style = Dummy()

import inspect
import os
import pprint

from collections import OrderedDict


def caller_str():
    frame, filename, line_number, function_name, lines, index = inspect.stack()[2]
    last_few_path_pieces = filename.split(os.path.sep)[-2:]
    reported_fn = os.path.join(*last_few_path_pieces)
    return "{}:{}:{}".format(reported_fn, function_name, line_number)


class Log(object):
    _verbosities = OrderedDict([
        ("debug", Fore.LIGHTCYAN_EX),
        ("info", Fore.LIGHTBLUE_EX),
        ("success", Fore.LIGHTGREEN_EX),
        ("warn", Fore.YELLOW),
        ("error", Fore.RED),
    ])
    _enabled = _verbosities.keys()

    @classmethod
    def set_enabled(cls, keys):
        for key in keys:
            assert key in cls._verbosities.keys(), "{} unknown".format(key)
        cls._enabled = keys

    @classmethod
    def set_verbosity(cls, level):
        assert level in cls._verbosities.keys(), "{} unknown".format(level)
        all_keys = cls._verbosities.keys()
        start = all_keys.index(level)
        cls.set_enabled(all_keys[start:])
        cls.debug("Enabling ", all_keys[start:])

    @classmethod
    def get_verbosities(cls):
        return cls._verbosities


def make_logger(verbosity_type, color):
    def new_logger(cls, *txt):
        if verbosity_type in cls._enabled:
            if len(txt) == 1 and not isinstance(txt[0], (str, unicode)):
                    out_str = out_str = pprint.pformat(txt, width=50)
            else:
                out_str = " ".join(map(str, txt))

            if '\n' in out_str:
                pre = '\n'
            else:
                pre = ""
            print("{}:{} {}{}{}".format(caller_str(), pre, color, out_str, Style.RESET_ALL))

    return new_logger


for verbosity, color in Log._verbosities.items():
    setattr(Log, verbosity, classmethod(make_logger(verbosity, color)))


if __name__ == '__main__':
    Log.error("error:", 0)
    Log.warn("warn:", 1)
    Log.success("success:", 2)
    Log.info("info:", 3)
    Log.debug("debug:", 4)
