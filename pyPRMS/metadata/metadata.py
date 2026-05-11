
from __future__ import annotations

import io
import pkgutil
import xml.etree.ElementTree as xmlET

from collections import defaultdict
from packaging.version import Version

from pyPRMS.prms_helpers import set_date
from pyPRMS.constants import MetaDataType, NEW_DTYPE, NEW_PARAM_DTYPE, NEW_PTYPE_TO_DTYPE, PRMS_VERSION
from ..base.console import get_console_instance

con = None


class MetaData(object):
    """Class to handle variable and parameter metadata"""

    # For each metadata type, define the outer element name for each variable in the XML file
    _OUTSIDE_ELEM = {'control': 'control_param',
                     'parameters': 'parameter',
                     'dimensions': 'dimension',
                     'variables': 'variable',
                     'data_file': 'variable',
                     'cbh': 'variable'}

    def __init__(self, version: str | Version = PRMS_VERSION,
                 verbose: bool = False):
        """Create a MetaData object by parsing PRMS XML metadata files.

        Loads and parses all metadata types (control, dimensions, parameters,
        variables, data_file, cbh) from the bundled XML files, filtering by
        the specified PRMS version.

        :param version: PRMS version string or Version object for filtering metadata
        :param verbose: Output additional debug information during parsing
        """

        global con
        con = get_console_instance()

        fcn_map = {'control': self.__control_to_dict,
                   'dimensions': self.__dimensions_to_dict,
                   'parameters': self._parameters_to_dict,
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
        """Return the complete metadata dictionary.

        :returns: Dictionary containing all parsed metadata keyed by type
            (info, control, dimensions, parameters, variables, data_file, cbh)
        """
        return self.__meta_dict

    def __repr__(self) -> str:
        """String representation of MetaData object.

        :returns: string with version and entry counts per metadata type
        """
        return f"MetaData(version='{self.__version}')"

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def __filter_by_version(self, elem: xmlET.Element, name: str,
                            meta_dict: dict, req_version: Version) -> bool:
        """Apply version and deprecation filtering to a metadata element.

        If the element passes filtering, an empty entry is created in *meta_dict*
        for *name*. If it fails, any partially-created entry is removed.

        :param elem: XML element to check
        :param name: Name of the variable/parameter
        :param meta_dict: dictionary being built (entry may be added/removed)
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
    def __find_text(elem: xmlET.Element, tag: str) -> str | None:
        """Find a child element and return its text content, or None if not found.

        :param elem: Parent XML element
        :param tag: Tag name of the child element to find
        :returns: Text content of the child element, or None if element is missing
        """
        child = elem.find(tag)
        return child.text if child is not None else None

    @staticmethod
    def __extract_common(elem: xmlET.Element, meta_entry: dict):
        """Extract dimensions, modules, and requires elements common to most metadata types.

        :param elem: XML element to extract from
        :param meta_entry: dictionary entry to populate
        """

        for cdim in elem.findall('./dimensions/dimension'):
            meta_entry['dimensions'].append(cdim.attrib.get('name'))

        for cmod in elem.findall('./modules/module'):
            meta_entry['modules'].append(cmod.text)

        for creq in elem.findall('./requires/*'):
            meta_entry[f'requires_{creq.tag}'].append(creq.text)

    @staticmethod
    def __extract_valid_values(elem: xmlET.Element, meta_entry: dict):
        """Extract valid values from an XML element.

        :param elem: XML element to extract from
        :param meta_entry: dictionary entry to populate
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
                          req_version: Version) -> dict:
        """Convert control variables metadata to dictionary.

        :param xml_root: XML root element
        :param meta_type: Type of metadata
        :param req_version: Required minimum version for filtering
        """

        meta_dict: dict = {}

        for elem in xml_root.findall(self._OUTSIDE_ELEM[meta_type]):
            name = elem.attrib.get('name')

            if self.__filter_by_version(elem, name, meta_dict, req_version):
                continue

            if name in ['start_time', 'end_time']:
                meta_dict[name]['datatype'] = 'datetime'
            else:
                type_text = self.__find_text(elem, 'type')
                datatype = int(type_text)
                meta_dict[name]['datatype'] = NEW_DTYPE[datatype]

            elems = {'description': 'desc',
                     'numvals': 'numvals',
                     'default': 'default', }

            for ek, ev in elems.items():
                text = self.__find_text(elem, ev)
                if text is None:
                    continue

                try:
                    if ev == 'numvals':
                        if text in ['1', '6']:
                            meta_dict[name]['context'] = 'scalar'
                        else:
                            meta_dict[name]['context'] = 'array'
                    elif ev == 'default':
                        cdtype = NEW_PTYPE_TO_DTYPE[meta_dict[name]['datatype']]

                        if meta_dict[name]['datatype'] == 'datetime':
                            meta_dict[name][ek] = cdtype(set_date(text))
                        else:
                            meta_dict[name][ek] = cdtype(text)
                    else:
                        meta_dict[name][ek] = text
                except ValueError:
                    meta_dict[name][ek] = text

            force_default_text = self.__find_text(elem, 'force_default')
            if force_default_text is not None:
                meta_dict[name]['force_default'] = force_default_text == '1'

            self.__extract_valid_values(elem, meta_dict[name])

        return meta_dict

    def _parameters_to_dict(self, xml_root: xmlET.Element,
                            meta_type: str,
                            req_version: Version) -> dict:
        """Convert parameter metadata to dictionary.

        :param xml_root: XML root element
        :param meta_type: Type of metadata
        :param req_version: Required minimum version for filtering
        """

        meta_dict: dict = {}

        for elem in xml_root.findall(self._OUTSIDE_ELEM[meta_type]):
            name = elem.attrib.get('name')

            if self.__filter_by_version(elem, name, meta_dict, req_version):
                continue

            # Convert to defaultdict for list-valued fields
            meta_dict[name] = defaultdict(list, meta_dict[name])

            datatype = self.__find_text(elem, 'type')
            meta_dict[name]['datatype'] = NEW_PARAM_DTYPE[datatype]

            elems = {'description': 'desc',
                     'help': 'help',
                     'units': 'units',
                     'default': 'default',
                     'minimum': 'minimum',
                     'maximum': 'maximum'}

            for ek, ev in elems.items():
                text = self.__find_text(elem, ev)

                if ek in ['default', 'minimum', 'maximum']:
                    # Try to convert to the parameter datatype
                    # Bounded parameters will fail
                    cdtype = NEW_PTYPE_TO_DTYPE[meta_dict[name]['datatype']]

                    if text is None:
                        meta_dict[name][ek] = ''
                    else:
                        try:
                            meta_dict[name][ek] = cdtype(text)
                        except ValueError:
                            if text == 'bounded':
                                meta_dict[name][ek] = meta_dict[name]['default']
                            else:
                                meta_dict[name][ek] = text
                else:
                    if text is not None:
                        meta_dict[name][ek] = text

            self.__extract_common(elem, meta_dict[name])
            self.__extract_valid_values(elem, meta_dict[name])

        return meta_dict

    def __dimensions_to_dict(self, xml_root: xmlET.Element,
                             meta_type: str,
                             req_version: Version) -> dict:
        """Convert dimensions metadata to dictionary.

        :param xml_root: XML root element
        :param meta_type: Type of metadata
        :param req_version: Required minimum version for filtering
        """

        meta_dict: dict = {}

        for elem in xml_root.findall(self._OUTSIDE_ELEM[meta_type]):
            name = elem.attrib.get('name')

            meta_dict[name] = {}

            elems = {'description': {'orig_name': 'desc',
                                     'datatype': str},
                     'size': {'orig_name': 'size',
                              'datatype': int},
                     'default': {'orig_name': 'default',
                                 'datatype': int},
                     'is_fixed': {'orig_name': 'is_fixed',
                                  'datatype': lambda x: x in ('1', 'True', 'true')}}

            for ek, ev in elems.items():
                text = self.__find_text(elem, ev['orig_name'])
                if text is not None:
                    meta_dict[name][ek] = ev['datatype'](text)
                elif ek == 'is_fixed':
                    meta_dict[name][ek] = False

            for creq in elem.findall('./requires/*'):
                meta_dict[name].setdefault(f'requires_{creq.tag}', list()).append(creq.text)

        return meta_dict

    def __variables_to_dict(self, xml_root: xmlET.Element,
                            meta_type: str,
                            req_version: Version) -> dict:
        """Convert output variables metadata to dictionary.

        :param xml_root: XML root element
        :param meta_type: Type of metadata
        :param req_version: Required minimum version for filtering
        """

        meta_dict: dict = {}

        for elem in xml_root.findall(self._OUTSIDE_ELEM[meta_type]):
            name = elem.attrib.get('name')

            meta_dict[name] = defaultdict(list)

            datatype = self.__find_text(elem, 'type')
            meta_dict[name]['datatype'] = NEW_PARAM_DTYPE[datatype]

            for ek, ev in {'description': 'desc', 'units': 'units'}.items():
                text = self.__find_text(elem, ev)
                if text is not None:
                    meta_dict[name][ek] = text

            self.__extract_common(elem, meta_dict[name])

        return meta_dict

    def __cbh_to_dict(self, xml_root: xmlET.Element,
                      meta_type: str,
                      req_version: Version) -> dict:
        """Convert CBH variables metadata to dictionary.

        :param xml_root: XML root element
        :param meta_type: Type of metadata
        :param req_version: Required minimum version for filtering
        """

        meta_dict: dict = {}

        for elem in xml_root.findall(self._OUTSIDE_ELEM[meta_type]):
            name = elem.attrib.get('name')

            if self.__filter_by_version(elem, name, meta_dict, req_version):
                continue

            # Convert to defaultdict for list-valued fields
            meta_dict[name] = defaultdict(list, meta_dict[name])

            datatype = self.__find_text(elem, 'type')
            meta_dict[name]['datatype'] = NEW_PARAM_DTYPE[datatype]

            for ek, ev in {'description': 'desc', 'help': 'help', 'units': 'units',
                           'default': 'default', 'minimum': 'minimum', 'maximum': 'maximum'}.items():
                text = self.__find_text(elem, ev)
                if text is not None:
                    meta_dict[name][ek] = text

            self.__extract_common(elem, meta_dict[name])

        return meta_dict

    def __data_file_to_dict(self, xml_root: xmlET.Element,
                            meta_type: str,
                            req_version: Version) -> dict:
        """Convert Data File variables metadata to dictionary.

        :param xml_root: XML root element
        :param meta_type: Type of metadata
        :param req_version: Required minimum version for filtering
        """

        meta_dict: dict = {}

        for elem in xml_root.findall(self._OUTSIDE_ELEM[meta_type]):
            name = elem.attrib.get('name')

            if self.__filter_by_version(elem, name, meta_dict, req_version):
                continue

            # Convert to defaultdict for list-valued fields
            meta_dict[name] = defaultdict(list, meta_dict[name])

            datatype = self.__find_text(elem, 'type')
            meta_dict[name]['datatype'] = NEW_PARAM_DTYPE[datatype]

            for ek, ev in {'description': 'desc', 'units': 'units',
                           'minimum': 'minimum', 'maximum': 'maximum'}.items():
                text = self.__find_text(elem, ev)
                if text is not None:
                    meta_dict[name][ek] = text

            self.__extract_common(elem, meta_dict[name])

        return meta_dict
