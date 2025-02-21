import io
import pkgutil
import xml.etree.ElementTree as xmlET   # type: ignore

from collections import defaultdict
from packaging.version import Version
from typing import Dict, Optional, Union

from pyPRMS.prms_helpers import set_date
from pyPRMS.constants import MetaDataType, NEW_PTYPE_TO_DTYPE, PRMS_VERSION

from rich.console import Console
from rich import pretty

pretty.install()
con = Console(record=False, force_jupyter=False)

# For each metadata type, define the outer element name for each variable in the XML file
outside_elem = {'control': 'control_param',
                'parameters': 'parameter',
                'dimensions': 'dimension',
                'variables': 'variable',
                'cbh': 'variable'}

NEW_DTYPE = {1: 'int32', 2: 'float32', 3: 'float64', 4: 'string'}
NEW_PARAM_DTYPE = {'I': 'int32', 'F': 'float32', 'D': 'float64', 'S': 'string'}


class MetaData(object):
    """Class to handle variable and parameter metadata"""

    def __init__(self, version: Union[str, Version] = PRMS_VERSION,
                 verbose: bool = False):
        # meta_type - one of control, dimension, parameter, output
        # version - PRMS major version to use for filtering

        fcn_map = {'control': self.__control_to_dict,
                   'dimensions': self.__dimensions_to_dict,
                   'parameters': self.__parameters_to_dict,
                   'variables': self.__variables_to_dict,
                   'cbh': self.__cbh_to_dict}

        self.__meta_dict: MetaDataType = {}

        if isinstance(version, str):
            version = Version(version)

        self.__version: Version = version
        self.__verbose = verbose

        # Add information about the metadata
        self.__meta_dict['info'] = {'version': str(self.__version)}

        if self.__verbose:
            con.print(f'[green]INFO[/]: Metadata for PRMS version {self.__version}')

        # meta_type: one of - control, dimensions, parameters, variables
        for mt, mf in fcn_map.items():
            xml_fh = io.StringIO(pkgutil.get_data('pyPRMS', f'xml/{mt}.xml').decode('utf-8'))
            xml_tree = xmlET.parse(xml_fh)
            xml_root = xml_tree.getroot()

            if self.__verbose:
                con.print(f'[bold green]{mt}[/bold green]')
            self.__meta_dict[mt] = mf(xml_root, mt, self.__version)

    @property
    def version(self) -> Version:
        """Return the PRMS version used for metadata selection

        :returns: PRMS version
        """

        return self.__version

    @property
    def metadata(self) -> MetaDataType:
        return self.__meta_dict

    def __control_to_dict(self, xml_root: xmlET.Element,
                          meta_type: str,
                          req_version: Version) -> Dict:
        """Convert control variables metadata to dictionary

        :param xml_root: XML root element
        :param meta_type: Type of metadata
        :param req_version: Required minimum version for filtering
        """

        meta_dict: Dict = {}

        for elem in xml_root.findall(outside_elem[meta_type]):
            name = elem.attrib.get('name')

            meta_dict[name] = {}
            try:
                var_version = Version(elem.attrib.get('version'))

                if var_version > req_version:
                    if self.__verbose:   # pragma: no cover
                        print(f'{name} rejected by version {str(var_version)}, req: {str(req_version)}')

                    del meta_dict[name]
                    continue
                meta_dict[name]['version'] = str(var_version)
            except TypeError:
                pass

            try:
                depr_version = Version(elem.attrib.get('deprecated'))

                if depr_version <= req_version:
                    if self.__verbose:   # pragma: no cover
                        print(f'{name} rejected by deprecation version {str(depr_version)}, req: {str(req_version)}')

                    del meta_dict[name]
                    continue
                meta_dict[name]['deprecated'] = depr_version
            except TypeError:
                pass

            if name in ['start_time', 'end_time']:
                meta_dict[name]['datatype'] = 'datetime'
            else:
                datatype = int(elem.find('type').text)
                meta_dict[name]['datatype'] = NEW_DTYPE[datatype]

            elems = {'description': 'desc',
                     'numvals': 'numvals',
                     'default': 'default', }

            for ek, ev in elems.items():
                try:
                    if ev == 'numvals':
                        tmp = elem.find(ev).text

                        if tmp in ['1', '6']:
                            meta_dict[name]['context'] = 'scalar'
                        else:
                            meta_dict[name]['context'] = 'array'
                        # meta_dict[name][ek] = int(elem.find(ev).text)
                    elif ev == 'default':
                        cdtype = NEW_PTYPE_TO_DTYPE[meta_dict[name]['datatype']]

                        if meta_dict[name]['datatype'] == 'datetime':
                            meta_dict[name][ek] = cdtype(set_date(elem.find(ev).text))
                        else:
                            meta_dict[name][ek] = cdtype(elem.find(ev).text)
                    else:
                        meta_dict[name][ek] = elem.find(ev).text
                except ValueError:
                    meta_dict[name][ek] = elem.find(ev).text
                except AttributeError:
                    pass

            # meta_dict[name]['description'] = elem.find('desc').text
            # meta_dict[name]['numvals'] = elem.find('numvals').text

            if elem.find('force_default') is not None:
                meta_dict[name]['force_default'] = elem.find('force_default').text == '1'

            # Possible valid values for variable
            # outvals = {}
            for cvals in elem.findall('./values'):
                meta_dict[name]['valid_value_type'] = cvals.attrib.get('type')

                meta_dict[name]['valid_values'] = {}
                for cv in cvals.findall('./value'):
                    meta_dict[name]['valid_values'][cv.attrib.get('name')] = cv.text

        return meta_dict

    def __parameters_to_dict(self, xml_root: xmlET.Element,
                             meta_type: str,
                             req_version: Version) -> Dict:
        """Convert parameter metadata to dictionary"""

        meta_dict: Dict = {}

        for elem in xml_root.findall(outside_elem[meta_type]):
            name = elem.attrib.get('name')

            meta_dict[name] = defaultdict(list)

            try:
                var_version = Version(elem.attrib.get('version'))

                if var_version > req_version:
                    if self.__verbose:   # pragma: no cover
                        print(f'{name} rejected by version {str(var_version)}, req: {str(req_version)}')

                    del meta_dict[name]
                    continue
                meta_dict[name]['version'] = str(var_version)
            except TypeError:
                pass

            try:
                depr_version = Version(elem.attrib.get('deprecated'))

                if depr_version <= req_version:
                    if self.__verbose:   # pragma: no cover
                        print(f'{name} rejected by deprecation version {str(depr_version)}, req: {str(req_version)}')

                    del meta_dict[name]
                    continue
                meta_dict[name]['deprecated'] = depr_version
            except TypeError:
                pass

            # var_version = version_info(elem.attrib.get('version'))
            # depr_version = version_info(elem.attrib.get('deprecated'))
            #
            # if var_version.major is not None and var_version.major > version.major:
            #     if self.__verbose:   # pragma: no cover
            #         print(f'{name} rejected by version')
            #     continue
            # if depr_version.major is not None and depr_version.major <= version.major:
            #     if self.__verbose:   # pragma: no cover
            #         print(f'{name} rejected by deprecation version')
            #     continue
            #
            # meta_dict[name] = defaultdict(list)
            #
            # var_version = elem.attrib.get('version')
            # if var_version is not None:
            #     meta_dict[name]['version'] = var_version
            #     # meta_dict[name]['version'] = elem.attrib.get('version')
            #
            # depr_version = elem.attrib.get('deprecated')
            # if depr_version is not None:
            #     meta_dict[name]['deprecated'] = depr_version

            datatype = elem.find('type').text
            meta_dict[name]['datatype'] = NEW_PARAM_DTYPE[datatype]

            elems = {'description': 'desc',
                     'help': 'help',
                     'units': 'units',
                     'default': 'default',
                     'minimum': 'minimum',
                     'maximum': 'maximum', }

            for ek, ev in elems.items():
                if ek in ['default', 'minimum', 'maximum']:
                    # Try to convert to the parameter datatype
                    # Bounded parameters will fail
                    cdtype = NEW_PTYPE_TO_DTYPE[meta_dict[name]['datatype']]

                    try:
                        meta_dict[name][ek] = cdtype(elem.find(ev).text)
                    except ValueError:
                        # Leave the value as a string
                        if elem.find(ev).text == 'bounded':
                            meta_dict[name][ek] = meta_dict[name]['default']
                        else:
                            meta_dict[name][ek] = elem.find(ev).text
                    except AttributeError:
                        # Occurs when element does not exist; just default to string
                        meta_dict[name][ek] = ''
                else:
                    try:
                        meta_dict[name][ek] = elem.find(ev).text
                    except AttributeError:
                        pass

            for cdim in elem.findall('./dimensions/dimension'):
                meta_dict[name]['dimensions'].append(cdim.attrib.get('name'))

            for cmod in elem.findall('./modules/module'):
                meta_dict[name]['modules'].append(cmod.text)

            for creq in elem.findall('./requires/*'):
                meta_dict[name][f'requires_{creq.tag}'].append(creq.text)

        return meta_dict

    def __dimensions_to_dict(self, xml_root: xmlET.Element,
                             meta_type: str,
                             req_version: Version) -> Dict:
        """Convert control variables metadata to dictionary"""

        meta_dict: Dict = {}

        for elem in xml_root.findall(outside_elem[meta_type]):
            name = elem.attrib.get('name')

            meta_dict[name] = {}

            elems = {'description': {'orig_name': 'desc',
                                     'datatype': str},
                     'size': {'orig_name': 'size',
                              'datatype': int},
                     'default': {'orig_name': 'default',
                                 'datatype': int},
                     'is_fixed': {'orig_name': 'is_fixed',
                                  'datatype': bool}}

            for ek, ev in elems.items():
                try:
                    meta_dict[name][ek] = ev['datatype'](elem.find(ev['orig_name']).text)
                except AttributeError:
                    if ek == 'is_fixed':
                        meta_dict[name][ek] = False
                    pass

            for creq in elem.findall('./requires/*'):
                meta_dict[name].setdefault(f'requires_{creq.tag}', list()).append(creq.text)

        return meta_dict

    def __variables_to_dict(self, xml_root: xmlET.Element,
                            meta_type: str,
                            req_version: Version) -> Dict:
        """Convert output variables metadata to dictionary"""

        meta_dict: Dict = {}

        for elem in xml_root.findall(outside_elem[meta_type]):
            name = elem.attrib.get('name')
            # var_version = version_info(elem.attrib.get('version'))
            # depr_version = version_info(elem.attrib.get('deprecated'))
            #
            # if var_version.major is not None and var_version.major > version:
            #     if self.__verbose:
            #         print(f'{name} rejected by version')
            #     continue
            # if depr_version.major is not None and depr_version.major <= version:
            #     if self.__verbose:
            #         print(f'{name} rejected by deprecation version')
            #     continue

            meta_dict[name] = defaultdict(list)

            # var_version = elem.attrib.get('version')
            # if var_version is not None:
            #     meta_dict[name]['version'] = var_version
            #     # meta_dict[name]['version'] = elem.attrib.get('version')
            #
            # depr_version = elem.attrib.get('deprecated')
            # if depr_version is not None:
            #     meta_dict[name]['deprecated'] = depr_version

            datatype = elem.find('type').text
            meta_dict[name]['datatype'] = NEW_PARAM_DTYPE[datatype]

            elems = {'description': 'desc',
                     'units': 'units', }

            for ek, ev in elems.items():
                try:
                    meta_dict[name][ek] = elem.find(ev).text
                except AttributeError:
                    pass

            for cdim in elem.findall('./dimensions/dimension'):
                meta_dict[name]['dimensions'].append(cdim.attrib.get('name'))

            for cmod in elem.findall('./modules/module'):
                meta_dict[name]['modules'].append(cmod.text)

            # for creq in elem.findall('./requires/*'):
            #     meta_dict[name][f'requires_{creq.tag}'].append(creq.text)

        return meta_dict

    def __cbh_to_dict(self, xml_root: xmlET.Element,
                      meta_type: str,
                      req_version: Version) -> Dict:
        """Convert cbh variables metadata to dictionary"""

        meta_dict: Dict = {}

        for elem in xml_root.findall(outside_elem[meta_type]):
            name = elem.attrib.get('name')

            meta_dict[name] = defaultdict(list)

            try:
                var_version = Version(elem.attrib.get('version'))

                if var_version > req_version:
                    if self.__verbose:   # pragma: no cover
                        print(f'{name} rejected by version {str(var_version)}, req: {str(req_version)}')

                    del meta_dict[name]
                    continue
                meta_dict[name]['version'] = str(var_version)
            except TypeError:
                pass

            try:
                depr_version = Version(elem.attrib.get('deprecated'))

                if depr_version <= req_version:
                    if self.__verbose:   # pragma: no cover
                        print(f'{name} rejected by deprecation version {str(depr_version)}, req: {str(req_version)}')

                    del meta_dict[name]
                    continue
                meta_dict[name]['deprecated'] = depr_version
            except TypeError:
                pass

            datatype = elem.find('type').text
            meta_dict[name]['datatype'] = NEW_PARAM_DTYPE[datatype]

            elems = {'description': 'desc',
                     'help': 'help',
                     'units': 'units',
                     'default': 'default',
                     'minimum': 'minimum',
                     'maximum': 'maximum', }

            for ek, ev in elems.items():
                try:
                    meta_dict[name][ek] = elem.find(ev).text
                except AttributeError:
                    pass

            for cdim in elem.findall('./dimensions/dimension'):
                meta_dict[name]['dimensions'].append(cdim.attrib.get('name'))

            for cmod in elem.findall('./modules/module'):
                meta_dict[name]['modules'].append(cmod.text)

            for creq in elem.findall('./requires/*'):
                meta_dict[name][f'requires_{creq.tag}'].append(creq.text)

        return meta_dict