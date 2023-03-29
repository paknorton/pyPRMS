import io
import pkgutil
import xml.etree.ElementTree as xmlET   # type: ignore

from pyPRMS.prms_helpers import version_info

outside_elem = {'control': 'control_param',
                'parameters': 'parameter',
                }

NEW_DTYPE = {1: 'int32', 2: 'float32', 3: 'float64', 4: 'string'}

class MetaData(object):
    """Class to handle variable and parameter metadata"""

    def __init__(self, meta_type='control'):
        fcn_map = {'control': self.__control_to_dict}

        self.__meta_dict = {}

        # meta_type: one of - control, dimensions, parameters, variables
        xml_fh = io.StringIO(pkgutil.get_data('pyPRMS', f'xml/{meta_type}.xml').decode('utf-8'))
        xml_tree = xmlET.parse(xml_fh)
        xml_root = xml_tree.getroot()

        # self.__meta_dict[meta_type] = self.__control_to_dict(xml_root, meta_type)
        self.__meta_dict[meta_type] = fcn_map[meta_type](xml_root, meta_type)

    @property
    def metadata(self):
        return self.__meta_dict

    @staticmethod
    def __control_to_dict(xml_root, meta_type):
        """Convert control file metadata to dictionary"""

        meta_dict = {}

        for elem in xml_root.findall(outside_elem[meta_type]):
            name = elem.attrib.get('name')
            meta_dict[name] = {}

            if elem.attrib.get('version') is not None:
                meta_dict[name]['version'] = elem.attrib.get('version')

            depr_version = elem.attrib.get('deprecated')
            if depr_version is not None:
                meta_dict[name]['deprecated'] = depr_version

            # version_info(elem.attrib.get('version'))
            # version_info(elem.attrib.get('deprecated'))

            # if (var_version.major is not None and var_version.major > version):
            #     if self.__verbose:
            #         print(f'{name} rejected by version')
            #     continue
            # if (depr_version.major is not None and depr_version.major <= version):
            #     if self.__verbose:
            #         print(f'{name} rejected by deprecation version')
            #     continue

            datatype = int(elem.find('type').text)
            meta_dict[name]['datatype'] = NEW_DTYPE[datatype]

            # TODO: this shouldn't be here
            # if name in ['start_time', 'end_time']:
            #     # Hack to handle PRMS approach to dates
            #     dt = elem.find('default').text.split('-')
            #     if len(dt) < 6:
            #         # pad short date with zeros for hms
            #         dt.extend(['0' for _ in range(6 - len(dt))])
            #     self.get(name).default = dt
            # else:
            #     # print('{}: {} {}'.format(name, type(elem.find('default').text), elem.find('default').text))
            #     self.get(name).default = elem.find('default').text
            meta_dict[name]['description'] = elem.find('desc').text

            if elem.find('force_default') is not None:
                meta_dict[name]['force_default'] = elem.find('force_default').text == '1'

            meta_dict[name]['numvals'] = elem.find('numvals').text

            # if name in ['start_time', 'end_time']:
            #     meta_dict[name]['context'] =

            # Possible valid values for variable
            outvals = {}
            for cvals in elem.findall('./values'):
                meta_dict[name]['valid_value_type'] = cvals.attrib.get('type')

                meta_dict[name]['valid_values'] = {}
                for cv in cvals.findall('./value'):
                    meta_dict[name]['valid_values'][cv.attrib.get('name')] = cv.text

        return meta_dict
