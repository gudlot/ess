import scipp as sc
import scippneutron as scn
import numpy as np
import h5py as h5
from shutil import copyfile
import sys


def fix_nexus_scipp(infile, outfile):
    """
    Currently there are some tweaks to make file working in scipp. This function won't be necessary in the future
    """

    copyfile(infile, outfile)
    with h5.File(outfile, 'r+') as f:

        group = f['entry/instrument']
        for monitor_name in filter(lambda k: k.startswith('monitor'), group):
            monitor_group = group[monitor_name]
            monitor_event_group = monitor_group[f'{monitor_name}_events']
            for key in list(monitor_event_group):
                monitor_group[key] = monitor_event_group.pop(key)
            del monitor_group[f'{monitor_name}_events']

def html_table(list_of_strings):
    html_file = open('report.html', 'w')
    html_file.write('<html>\n')
    html_file.write('<head>\n')
    html_file.write('<style>\n')
    html_file.write('table, th, td { border: 1px solid black; border - collapse: collapse;}\n')
    html_file.write('</style>\n')
    html_file.write('</head>\n')
    html_file.write('<table>\n')
    for sublist in list_of_strings:
        html_file.write('<tr>\n')
        for column in sublist:
            html_file.write(f'<td>{column}</td>\n')
        html_file.write('</tr>\n')
    html_file.write('</table>\n')
    html_file.close()


if __name__ == "__main__":

    files = ['60247-2022-02-28_2215.nxs',
     '60248-2022-02-28_2215.nxs',
     '60249-2022-02-28_2215.nxs',
     '60250-2022-02-28_2215.nxs',
     '60251-2022-02-28_2215.nxs',
     '60252-2022-02-28_2215.nxs',
     '60253-2022-02-28_2215.nxs',
     '60254-2022-02-28_2215.nxs',
     '60255-2022-02-28_2215.nxs',
     '60256-2022-02-28_2215.nxs',
     '60257-2022-02-28_2215.nxs',
     '60258-2022-02-28_2215.nxs',
     '60259-2022-02-28_2215.nxs',
     '60260-2022-02-28_2215.nxs',
     '60261-2022-02-28_2215.nxs',
     '60262-2022-02-28_2215.nxs',
     '60263-2022-02-28_2215.nxs',
     '60264-2022-02-28_2215.nxs',
     '60265-2022-02-28_2215.nxs',
     '60266-2022-02-28_2215.nxs',
     '60287-2022-02-28_2215.nxs',
     '60288-2022-02-28_2215.nxs',
     '60289-2022-02-28_2215.nxs',
     '60290-2022-02-28_2215.nxs',
     '60291-2022-02-28_2215.nxs',
     '60292-2022-02-28_2215.nxs',
     '60293-2022-02-28_2215.nxs',
     '60294-2022-02-28_2215.nxs',
     '60295-2022-02-28_2215.nxs',
     '60296-2022-02-28_2215.nxs',
     '60297-2022-02-28_2215.nxs',
     '60298-2022-02-28_2215.nxs',
     '60299-2022-02-28_2215.nxs',
     '60300-2022-02-28_2215.nxs',
     '60301-2022-02-28_2215.nxs']

    files = ['60300-2022-02-28_2215.nxs',
            '60301-2022-02-28_2215.nxs',
            '60303-2022-02-28_2215.nxs',
            '60304-2022-02-28_2215.nxs',
            '60305-2022-02-28_2215.nxs',
            '60306-2022-02-28_2215.nxs',
            '60307-2022-02-28_2215.nxs',
            '60308-2022-02-28_2215.nxs',
            '60309-2022-02-28_2215.nxs',
            '60310-2022-02-28_2215.nxs',
            '60311-2022-02-28_2215.nxs',
            '60312-2022-02-28_2215.nxs',
            '60313-2022-02-28_2215.nxs',
            '60314-2022-02-28_2215.nxs',
            '60315-2022-02-28_2215.nxs',
            '60316-2022-02-28_2215.nxs',
            '60317-2022-02-28_2215.nxs',
            '60318-2022-02-28_2215.nxs',
            '60319-2022-02-28_2215.nxs',
            '60320-2022-02-28_2215.nxs',
            '60321-2022-02-28_2215.nxs',
            '60322-2022-02-28_2215.nxs',
            '60323-2022-02-28_2215.nxs',
            '60324-2022-02-28_2215.nxs',
            '60325-2022-02-28_2215.nxs',
            '60326-2022-02-28_2215.nxs',
            '60327-2022-02-28_2215.nxs',
            '60328-2022-02-28_2215.nxs',
            '60329-2022-02-28_2215.nxs',
            '60330-2022-02-28_2215.nxs',
            #'60331-2022-02-28_2215.nxs',
            '60332-2022-02-28_2215.nxs',
            '60333-2022-02-28_2215.nxs',
            '60334-2022-02-28_2215.nxs',
            '60335-2022-02-28_2215.nxs',
            '60336-2022-02-28_2215.nxs',
            '60337-2022-02-28_2215.nxs',
            '60338-2022-02-28_2215.nxs',
            '60339-2022-02-28_2215.nxs',
            '60340-2022-02-28_2215.nxs',
            '60341-2022-02-28_2215.nxs',
            '60342-2022-02-28_2215.nxs',
            '60343-2022-02-28_2215.nxs',
            '60344-2022-02-28_2215.nxs',
            '60345-2022-02-28_2215.nxs',
            '60346-2022-02-28_2215.nxs',
            '60347-2022-02-28_2215.nxs',
            '60348-2022-02-28_2215.nxs',
            '60349-2022-02-28_2215.nxs',
            '60350-2022-02-28_2215.nxs',
            '60351-2022-02-28_2215.nxs',
            '60352-2022-02-28_2215.nxs',
            '60353-2022-02-28_2215.nxs',
            '60354-2022-02-28_2215.nxs',
            '60355-2022-02-28_2215.nxs',
            '60356-2022-02-28_2215.nxs',
            '60357-2022-02-28_2215.nxs',
            '60358-2022-02-28_2215.nxs',
            '60359-2022-02-28_2215.nxs',
            '60360-2022-02-28_2215.nxs',
            '60361-2022-02-28_2215.nxs',
            '60362-2022-02-28_2215.nxs',
            '60363-2022-02-28_2215.nxs',
            '60364-2022-02-28_2215.nxs',
            '60365-2022-02-28_2215.nxs',
            '60366-2022-02-28_2215.nxs',
            '60367-2022-02-28_2215.nxs',
            '60368-2022-02-28_2215.nxs',
            '60369-2022-02-28_2215.nxs',
            '60370-2022-02-28_2215.nxs',
            '60371-2022-02-28_2215.nxs',
            '60372-2022-02-28_2215.nxs',
            '60373-2022-02-28_2215.nxs',
            '60374-2022-02-28_2215.nxs',
            '60375-2022-02-28_2215.nxs',
            '60376-2022-02-28_2215.nxs',
            '60377-2022-02-28_2215.nxs',
            '60378-2022-02-28_2215.nxs',
            '60379-2022-02-28_2215.nxs',
            '60380-2022-02-28_2215.nxs',
            '60381-2022-02-28_2215.nxs',
            '60382-2022-02-28_2215.nxs',
            '60383-2022-02-28_2215.nxs',
            '60384-2022-02-28_2215.nxs',
            '60385-2022-02-28_2215.nxs',
            '60386-2022-02-28_2215.nxs',
            '60387-2022-02-28_2215.nxs',
            '60388-2022-02-28_2215.nxs',
            '60389-2022-02-28_2215.nxs',
            '60390-2022-02-28_2215.nxs',
            '60391-2022-02-28_2215.nxs',
            '60392-2022-02-28_2215.nxs',
            '60393-2022-02-28_2215.nxs',
            '60394-2022-02-28_2215.nxs',
            '60395-2022-02-28_2215.nxs']

    # files = ['60307-2022-02-28_2215.nxs',
    #         '60312-2022-02-28_2215.nxs',
    #         '60330-2022-02-28_2215.nxs',
    #         #'60331-2022-02-28_2215.nxs',
    #         '60335-2022-02-28_2215.nxs',
    #         '60356-2022-02-28_2215.nxs',
    #         '60364-2022-02-28_2215.nxs',
    #         '60365-2022-02-28_2215.nxs',
    #         '60367-2022-02-28_2215.nxs',
    #         '60375-2022-02-28_2215.nxs',
    #         '60395-2022-02-28_2215.nxs']

    html_strings = [['File', 'Events', 'TOF', 'Detector', 'M1', 'M2']]
    for input_file in files:

        html_row = [input_file]
        fixed_file = f'{input_file[:-4]}_fixed.nxs'
        fix_nexus_scipp(input_file, fixed_file)
        data = scn.load_nexus(data_file=fixed_file)

        no_events = len(data.bins.constituents['data'].values)
        html_row.append(no_events)
        detector_image = f'{input_file}_data_plot.png'

        scn.instrument_view(data, pixel_size=0.01, norm="log")

        tof_edges = sc.linspace(dim='tof', start=data.coords['tof'][0].values, stop=data.coords['tof'][-1].values,
                                num=100, unit='ns')
        histogrammed = sc.histogram(data, bins=tof_edges)
        histogrammed.sum('detector_id').plot(filename=detector_image)

        start_tof = data.coords['tof'][0].values
        end_tof = data.coords['tof'][-1].values

        print(f'File: {input_file} min_tof: {start_tof} max_tof: {end_tof}')
        html_row.append(f'{start_tof}-{end_tof}')

        html_row.append(f'<img src={detector_image}>')

        monitors = {
            'sample': {'incident': data.attrs["monitor_1"].value,
                       'transmission': data.attrs["monitor_2"].value}
        }

        monitors['sample']['incident'].coords['position'] = sc.vector(value=np.array([0, 0, 0.0]), unit=sc.units.m)
        monitors['sample']['incident'].coords['source_position'] = sc.vector(value=np.array([0, 0, -25.3]),
                                                                             unit=sc.units.m)
        monitors['sample']['transmission'].coords['position'] = sc.vector(value=np.array([0, 0, 0.0]), unit=sc.units.m)
        monitors['sample']['transmission'].coords['source_position'] = sc.vector(value=np.array([0, 0, -25.3]),
                                                                                 unit=sc.units.m)

        tof_edges = sc.linspace(dim='tof',start=data.coords['tof'][0].values, stop=data.coords['tof'][-1].values, num=100, unit='ns')
        histogrammed_monitor1 = sc.histogram(monitors['sample']['incident'], bins=tof_edges)
        histogrammed_monitor2 = sc.histogram(monitors['sample']['transmission'], bins=tof_edges)

        m1_image = f'{input_file}_m1_plot.png'
        m2_image = f'{input_file}_m2_plot.png'

        histogrammed_monitor1.sum('detector_id').plot(filename=m1_image)
        histogrammed_monitor2.sum('detector_id').plot(filename=m2_image)
        html_row.append(f'<img src={m1_image}>')
        html_row.append(f'<img src={m2_image}>')

        html_strings.append(html_row)

    html_table(html_strings)