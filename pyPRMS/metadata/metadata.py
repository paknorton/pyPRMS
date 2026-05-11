
import io
import pkgutil
import xml.etree.ElementTree as xmlET   # type: ignore

from collections import defaultdict
from packaging.version import Version
from typing import Dict, Optional, Union

from pyPRMS.prms_helpers import set_date
from pyPRMS.constants import MetaDataType, NEW_PTYPE_TO_DTYPE, PRMS_VERSION
from ..base.console import get_console_instance

con = None

# For each metadata type, define the outer element name for each variable in the XML file
outside_elem = {'control': 'control_param',
                'parameters': 'parameter',
                'dimensions': 'dimension',
                'variables': 'variable',
                'data_file': 'variable',
                'cbh': 'variable'}

NEW_DTYPE = {1: 'int32', 2: 'float32', 3: 'float64', 4: 'string'}
NEW_PARAM_DTYPE = {'I': 'int32', 'F': 'float32', 'D': 'float64', 'S': 'string'}


class MetaData(object):
    """Class to handle variable and parameter metadata"""

    def __init__(self, version: Union[str, Version] = PRMS_VERSION,
                 verbose: bool = False):
        # meta_type - one of control, dimension, parameter, output
        # version - PRMS major version to use for filtering

        global con
        con = get_console_instance()

        fcn_map = {'control': self.__control_to_dict,
                   'dimensions': self.__dimensions_to_dict,
                   'parameters': self.__parameters_to_dict,
                   'variables': self.__variables_to_dict,
                   'data_file': self.__data_file_to_dict,
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

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def __filter_by_version(self, elem: xmlET.Element, name: str,
                            meta_dict: Dict, req_version: Version) -> bool:
        """Apply version and deprecation filtering to a metadata element.

        If the element passes filtering, an empty entry is created in *meta_dict*
        for *name*. If it fails, any partially-created entry is removed.

        :param elem: XML element to check
        :param name: Name of the variable/parameter
        :param meta_dict: Dictionary being built (entry may be added/removed)
        :param req_version: Required PRMS version for filtering
        :returns: True if the element should be skipped, False if it passes
        """

        meta_dict[name] = {}

        try:
            var_version = Version(elem.attrib.get('version'))

            if var_version > req_version:
                if self.__verbose:   # pragma: no cover
                    con.print(f'[green]INFO[/]: [bold]{name}[/] requires version {str(var_version)}')
                del meta_dict[name]
                return True
            meta_dict[name]['version'] = str(var_version)
        except TypeError:
            pass

        try:
            depr_version = Version(elem.attrib.get('deprecated'))

            if depr_version <= req_version:
                if self.__verbose:   # pragma: no cover
                    con.print(f'[green]INFO[/]: [bold]{name}[/] was deprecated at version {str(depr_version)}')
                del meta_dict[name]
                return True
            meta_dict[name]['deprecated'] = depr_version
        except TypeError:
            pass

        return False

    @staticmethod
    def __extract_common(elem: xmlET.Element, meta_entry: Dict):
        """Extract dimensions, modules, and requires elements common to most metadata types.

        :param elem: XML element to extract from
        :param meta_entry: Dictionary entry to populate
        """

        for cdim in elem.findall('./dimensions/dimension'):
            meta_entry['dimensions'].append(cdim.attrib.get('name'))

        for cmod in elem.findall('./modules/module'):
            meta_entry['modules'].append(cmod.text)

        for creq in elem.findall('./requires/*'):
            meta_entry[f'requires_{creq.tag}'].append(creq.text)

    @staticmethod
    def __extract_valid_values(elem: xmlET.Element, meta_entry: Dict):
        """Extract valid values from an XML element.

        :param elem: XML element to extract from
        :param meta_entry: Dictionary entry to populate
        """

        for cvals in elem.findall('./values'):
            meta_entry['valid_value_type'] = cvals.attrib.get('type')

            meta_entry['valid_values'] = {}
            for cv in cvals.findall('./value'):
                meta_entry['valid_values'][cv.attrib.get('name')] = cv.text

    # ------------------------------------------------------------------
    # Per-type parsing methods
    # ------------------------------------------------------------------

    def __control_to_dict(self, xml_root: xmlET.Element,
                          meta_type: str,
                          req_version: Version) -> Dict:
        """Convert control variables metadata to dictionary.

        :param xml_root: XML root element
        :param meta_type: Type of metadata
        :param req_version: Required minimum version for filtering
        """

        meta_dict: Dict = {}

        for elem in xml_root.findall(outside_elem[meta_type]):
            name = elem.attrib.get('name')

            if self.__filter_by_version(elem, name, meta_dict, req_version):
                continue

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

            if elem.find('force_default') is not None:
                meta_dict[name]['force_default'] = elem.find('force_default').text == '1'

            self.__extract_valid_values(elem, meta_dict[name])

        return meta_dict

    def __parameters_to_dict(self, xml_root: xmlET.Element,
                             meta_type: str,
                             req_version: Version) -> Dict:
        """Convert parameter metadata to dictionary.

        :param xml_root: XML root element
        :param meta_type: Type of metadata
        :param req_version: Required minimum version for filtering
        """

        meta_dict: Dict = {}

        for elem in xml_root.findall(outside_elem[meta_type]):
            name = elem.attrib.get('name')

            if self.__filter_by_version(elem, name, meta_dict, req_version):
                continue

            # Convert to defaultdict for list-valued fields
            meta_dict[name] = defaultdict(list, meta_dict[name])

            datatype = elem.find('type').text
            meta_dict[name]['datatype'] = NEW_PARAM_DTYPE[datatype]

            elems = {'description': 'desc',
                     'help': 'help',
                     'units': 'units',
                     'default': 'default',
                     'minimum': 'minimum',
                     'maximum': 'maximum'}

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

            self.__extract_common(elem, meta_dict[name])
            self.__extract_valid_values(elem, meta_dict[name])

        return meta_dict

    def __dimensions_to_dict(self, xml_root: xmlET.Element,
                             meta_type: str,
                             req_version: Version) -> Dict:
        """Convert dimensions metadata to dictionary.

        :param xml_root: XML root element
        :param meta_type: Type of metadata
        :param req_version: Required minimum version for filtering
        """

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
        """Convert output variables metadata to dictionary.

        :param xml_root: XML root element
        :param meta_type: Type of metadata
        :param req_version: Required minimum version for filtering
        """

        meta_dict: Dict = {}

        for elem in xml_root.findall(outside_elem[meta_type]):
            name = elem.attrib.get('name')

            meta_dict[name] = defaultdict(list)

            datatype = elem.find('type').text
            meta_dict[name]['datatype'] = NEW_PARAM_DTYPE[datatype]

            for ek, ev in {'description': 'desc', 'units': 'units'}.items():
                try:
                    meta_dict[name][ek] = elem.find(ev).text
                except AttributeError:
                    pass

            self.__extract_common(elem, meta_dict[name])

        return meta_dict

    def __cbh_to_dict(self, xml_root: xmlET.Element,
                      meta_type: str,
                      req_version: Version) -> Dict:
        """Convert CBH variables metadata to dictionary.

        :param xml_root: XML root element
        :param meta_type: Type of metadata
        :param req_version: Required minimum version for filtering
        """

        meta_dict: Dict = {}

        for elem in xml_root.findall(outside_elem[meta_type]):
            name = elem.attrib.get('name')

            if self.__filter_by_version(elem, name, meta_dict, req_version):
                continue

            # Convert to defaultdict for list-valued fields
            meta_dict[name] = defaultdict(list, meta_dict[name])

            datatype = elem.find('type').text
            meta_dict[name]['datatype'] = NEW_PARAM_DTYPE[datatype]

            for ek, ev in {'description': 'desc', 'help': 'help', 'units': 'units',
                           'default': 'default', 'minimum': 'minimum', 'maximum': 'maximum'}.items():
                try:
                    meta_dict[name][ek] = elem.find(ev).text
                except AttributeError:
                    pass

            self.__extract_common(elem, meta_dict[name])

        return meta_dict

    def __data_file_to_dict(self, xml_root: xmlET.Element,
                            meta_type: str,
                            req_version: Version) -> Dict:
        """Convert Data File variables metadata to dictionary.

        :param xml_root: XML root element
        :param meta_type: Type of metadata
        :param req_version: Required minimum version for filtering
        """

        meta_dict: Dict = {}

        for elem in xml_root.findall(outside_elem[meta_type]):
            name = elem.attrib.get('name')

            if self.__filter_by_version(elem, name, meta_dict, req_version):
                continue

            # Convert to defaultdict for list-valued fields
            meta_dict[name] = defaultdict(list, meta_dict[name])

            datatype = elem.find('type').text
            meta_dict[name]['datatype'] = NEW_PARAM_DTYPE[datatype]

            for ek, ev in {'description': 'desc', 'units': 'units',
                           'minimum': 'minimum', 'maximum': 'maximum'}.items():
                try:
                    meta_dict[name][ek] = elem.find(ev).text
                except AttributeError:
                    pass

            self.__extract_common(elem, meta_dict[name])

        return meta_dict
