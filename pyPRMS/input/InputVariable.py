import pandas as pd   # type: ignore

from typing import Optional, Union


class InputVariable(object):
    """Class for working with input variables."""

    def __init__(self, name: str,
                 data: pd.DataFrame,
                 metadata: dict,
                 file_units: Optional[str] = None):
        """Initialize the InputVariable object.

        :param name: Name or kind of the input variable
        :param data: Input variable data
        :param metadata: Metadata for the input data variable
        :param file_units: Units of the input variable from the data file
        """

        self.__name = name
        self.__file_units = file_units
        self.data = data

        if 'data_file' in metadata:
            self.metadata = metadata['data_file'][name]
        else:
            self.metadata = metadata[name]

    def __str__(self) -> str:
        """Pretty-print string representation of the data file input variable information.

        :return: Pretty-print string of data file input variable information
        """

        outstr = f'----- PRMS data file input variable -----\n'
        outstr += f'name: {self.name}\n'

        for kk, vv in self.metadata.items():
            outstr += f'{kk}: {vv}\n'

        outstr += '----------\n'
        outstr += f'Number of rows: {len(self.data)}\n'
        outstr += f'Number of columns: {len(self.data.columns)}\n'

        return outstr

    @property
    def data(self) -> pd.DataFrame:
        """Returns the input variable data.

        :returns: Input variable dataframe
        """

        return self.__data

    @data.setter
    def data(self, data_in: pd.DataFrame):
        """Set the input variable data.

        :param data_in: Input variable data
        """

        col_names = {}
        for xx in data_in.columns:
            col_names[xx] = xx.split('_')[1]

        self.__data = data_in.copy()
        self.__data.rename(columns=col_names, inplace=True)

    @property
    def name(self) -> str:
        """Returns the input variable kind.

        :returns: Input variable kind
        """

        return self.__name

    @property
    def file_units(self) -> Union[str, None]:
        """Returns the input variable units.

        :returns: Input variable units
        """

        return self.__file_units
