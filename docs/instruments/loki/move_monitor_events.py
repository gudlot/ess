"""
Move monitor event data into the monitor group.

This is needed to load the file using scippneutron.load_nexus
with scippneutron v0.4.2
"""

from shutil import copyfile
import sys

import h5py as h5


def main():
    if len(sys.argv) != 3:
        print('Usage: move_monitor_events.py INFILE OUTFILE')
    infile = sys.argv[1]
    outfile = sys.argv[2]

    copyfile(infile, outfile)
    with h5.File(outfile, 'r+') as f:
        f['entry/instrument/name'] = 'LARMOR'
        group = f['entry/instrument']
        for monitor_name in filter(lambda k: k.startswith('monitor'), group):
            monitor_group = group[monitor_name]
            monitor_event_group = monitor_group[f'{monitor_name}_events']
            for key in list(monitor_event_group):
                monitor_group[key] = monitor_event_group.pop(key)
            del monitor_group[f'{monitor_name}_events']


if __name__ == '__main__':
    main()
