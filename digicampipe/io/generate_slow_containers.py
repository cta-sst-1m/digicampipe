import os
import re
import sys
import pkg_resources
from glob import glob

from astropy.io import fits


def unit_parser(column):
    unit = str(column.unit)
    if ',' in unit:
        unit = 'None'
    m = re.match('(?:\[)?(?P<unit>[\w]+)(?:\])?', unit)
    unit = m.group("unit")
    if unit.lower() == 'lsb' or unit.lower() == 'boolean':
        return 'bit'
    if unit.lower() == 'sec':
        return 'second'
    if unit.lower() == 'days':
        return 'day'
    if unit.lower() == 'celsius':
        return 'Celsius'
    return unit

def type_parser(column):
    if len(column.dtype.shape):
        return 'ndarray'
    return column.dtype.name


def description_parser(comment):
    ret = {}
    for line in comment.replace('\n', '').split('//'):
        if len(line) > 0:
            tmp = line.split(' - ')
            ret[tmp[0]] = tmp[1]
    return ret


def columns_info(hdu):
    res = {}
    descriptions = None
    if 'COMMENT' in hdu.header.keys():
        comment = hdu.header['COMMENT'].__str__()
        descriptions = description_parser(comment)
    for column in hdu.columns:
        description = column.name
        if descriptions is not None and column.name in descriptions:
            description = descriptions[column.name]
        res[column.name] = {
            'type': type_parser(column),
            'units': unit_parser(column),
            'desc': description
        }
    return res


def get_class_name(path):
    filename = os.path.basename(path)
    m = re.match('(?:slow_)?(?P<class>[\w]+?)_[\d\_]+\.fits', filename)
    class_name = m.group("class")
    return class_name


def generate_container_class(path_file, file=sys.stdout):
    class_name = get_class_name(path_file)
    print('class %sContainer(Container):' % class_name, file=file)
    with fits.open(path_file) as hdul:
        results = columns_info(hdul[1])
        lower_dict = dict((k.lower(), v) for k, v in results.items())
        if 'timestamp' not in lower_dict.keys():
            print('Warning in generate_container_class: no timestamp column found for class',
                  class_name, 'in file', path_file, file=file)
        for key in sorted(lower_dict):
            values = lower_dict[key]
            if str(values['units']) not in ['none', 'None', None]:
                data_printed = (key, values['type'], values['desc'], values['units'])
                print('    %s = Field(%s, "%s", unit=u.%s)' % data_printed, file=file)
            else:
                print('    %s = Field(%s, "%s")' % (key, values['type'], values['desc']), file=file)
    print('\n', file=file)
    return class_name


def generate_container_filler(path_file, file=sys.stdout):
    class_name = get_class_name(path_file)
    print('def fill_%s(event, hdu, slow_event):' % class_name, file=file)
    with fits.open(path_file) as hdul:
        results = columns_info(hdul[1])
        for key in sorted(results):
            print('    event.slow_data.%s.%s = hdu.data["%s"][slow_event]' % (class_name.lower(), key.lower(), key), file=file)
        print('    return event', file=file)
    print('\n', file=file)


def generate_fill_slow(filenames, file=sys.stdout):
    print('def fill_slow(class_name, event, hdu, slow_event):', file=file)
    first = True
    for path_file in filenames:
        class_name = get_class_name(path_file)
        if first:
            print('    if class_name == "%s":' % class_name, file=file)
            first = False
        else:
            print('    elif class_name == "%s":' % class_name, file=file)
        print('        event = fill_%s(event, hdu, slow_event)' % class_name, file=file)
    print('    else:', file=file)
    print('        print("ERROR in fill_slow(): class %s not known." %class_name)', file=file)
    print('        print("Try to regenerate slow data containers ?")', file=file)
    print('    return event', file=file)
    print('\n', file=file)


def generate_header(file=sys.stdout):
    print('from numpy import ndarray, uint8, int64, int32, float32, float64', file=file)
    print('', file=file)
    print('from astropy import units as u', file=file)
    print('from ctapipe.core import Container', file=file)
    print('try:', file=file)
    print('    from ctapipe.core import Field', file=file)
    print('except ImportError:', file=file)
    print('    from ctapipe.core import Item as Field', file=file)
    print('\n', file=file)


def generate_slow_container(filenames, file=sys.stdout):
    print('class SlowDataContainer(Container):', file=file)
    for path_file in filenames:
        class_name = get_class_name(path_file)
        print('    %s = Field(%sContainer(), "%s")' %(class_name.lower(), class_name, class_name), file=file)
    print('\n', file=file)


def generate_class_list(filenames, file=sys.stdout):
    print('__all__ = [', file=file)
    for path_file in filenames:
        class_name = get_class_name(path_file)
        print('    "%sContainer",' % class_name, file=file)
        print('    "fill_%s",' % class_name, file=file)
    print('    "SlowDataContainer",', file=file)
    print('    "fill_slow",', file=file)
    print(']\n\n', file=file)


if __name__ == '__main__':
    filenames = glob(
        pkg_resources.resource_filename(
            'digicampipe',
            'tests/resources/slow/*.fits'
        )
    )

    # get list of files to treat (to not treat classes more than once).
    files_to_process = []
    classes_to_process = []
    for path_file in filenames:
        class_name = get_class_name(path_file)
        if class_name in classes_to_process:
            continue
        classes_to_process.append(class_name)
        files_to_process.append(path_file)
    # create file
    python_file = open('./slow_container.py', 'w')
    #    python_file = sys.stdout
    generate_header(file=python_file)
    generate_class_list(files_to_process, file=python_file)
    for path_file in files_to_process:
        generate_container_class(path_file, file=python_file)
        generate_container_filler(path_file, file=python_file)
    generate_slow_container(files_to_process, file=python_file)
    generate_fill_slow(files_to_process, file=python_file)
