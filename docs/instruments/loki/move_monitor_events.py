"""
Move monitor event data into the monitor group.

This is needed to load the file using scippneutron.load_nexus
with scippneutron v0.4.2
"""

from shutil import copyfile
import sys
import numpy as np
import h5py as h5


def main():
    if len(sys.argv) != 3:
        print('Usage: move_monitor_events.py INFILE OUTFILE')
    infile = sys.argv[1]
    outfile = sys.argv[2]

    copyfile(infile, outfile)
    with h5.File(outfile, 'r+') as f:
        f['entry/instrument/name'] = 'LARMOR'
        
        #For tweaking Mantid
        group_entry = f['entry']
        nx_class = group_entry.create_group('sample')
        nx_class.attrs["NX_class"] = 'NXsample'
       
        #offsets = [14336]
        #for i in range(802816):
        #    f['entry']['instrument']['larmor_detector']['position'][i] -= 802816
        ##Adding detector_id
        #f['entry']['instrument']['monitor_1']['detector_id'] = np.array(401409)
        #f['entry']['instrument']['monitor_2']['detector_id'] = np.array(401410)
        group = f['entry/instrument']
        for monitor_name in filter(lambda k: k.startswith('monitor'), group):
            monitor_group = group[monitor_name]
            monitor_event_group = monitor_group[f'{monitor_name}_events']
            for key in list(monitor_event_group):
                monitor_group[key] = monitor_event_group.pop(key)
            del monitor_group[f'{monitor_name}_events']


if __name__ == '__main__':
    main()
