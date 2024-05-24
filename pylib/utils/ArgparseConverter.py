import argparse

import numpy as np
import torch


class ArgparseConverter():
    @staticmethod
    def str2None(v):
        """
        Helper function for argparser, converts str to None if str == 'none'

        Returns the string otherwise.

        Inputs:
            v: string
        """
        if v is None:
            return v
        elif isinstance(v, str):
            if v.lower() == 'none':
                return None
            else:
                return v
        else:
            raise argparse.ArgumentTypeError('String value expected.')

    @staticmethod
    def str2int(v):
        """
        Helper function for argparser, converts str to int if possible.

        Inputs:
            v: string
        """
        if v is None:
            return v
        elif isinstance(v, str):
            if v.lower() == 'none':
                return None
            elif v.lower() == 'nan':
                return np.NaN
            elif v.isnumeric():
                return int(v)
            elif v[0] == '-' and v[1:].isnumeric():
                return int(v)
            else:
                raise argparse.ArgumentTypeError(f'none or int expected not {v}')
        else:
            raise argparse.ArgumentTypeError('String value expected.')

    @staticmethod
    def str2float(v):
        """
        Helper function for argparser, converts str to float if possible.

        Inputs:
            v: string
        """
        if v is None:
            return v
        elif isinstance(v, str):
            if v.lower() == 'none':
                return None
            elif v.replace('.', '').replace('-', '').isnumeric():
                return float(v)
            elif 'e-' in v:
                return float(v)
            else:
                raise argparse.ArgumentTypeError(f'none or float expected not {v}')
        else:
            raise argparse.ArgumentTypeError('String value expected.')

    @staticmethod
    def str2bool(v):
        """
        Helper function for argparser, converts str to boolean for various string inputs.
        https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

        Inputs:
            v: string
        """
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def is_float(v) -> bool:
        try:
            float(v)
            return True
        except ValueError:
            return False

    @staticmethod
    def str2list(v, sep = '-'):
        """
        Helper function for argparser, converts str to list by splitting on sep.
        Empty string will be mapped to -1.

        Example for sep = '-': "-i-j-k" -> [-1, i, j, k]

        Inputs:
            v: string
            sep: separator
        """
        if v is None:
            return None
        elif isinstance(v, str):
            if v.lower() == 'none':
                return None
            elif v.lower() == 'empty':
                return []
            else:
                result = [i for i in v.split(sep)]
                for i, val in enumerate(result):
                    if val.isnumeric():
                        result[i] = int(val)
                    elif ArgparseConverter.is_float(val):
                        result[i] = float(val)
                    elif val == '':
                        result[i] = -1
                return result
        else:
            raise argparse.ArgumentTypeError('str value expected.')

    @staticmethod
    def str2list2D(v, sep1 = '\\', sep2 = '&'):
        """
        Helper function for argparser, converts str to list by splitting on sep1, then on sep2.

        Example for sep1 = '\\', sep2 = '&': "i & j \\ k & l" -> [[i, j], [k, l]]

        Inputs:
            v: string (any spaces will be ignored)
            sep: separator
        """
        if v is None:
            return None
        elif isinstance(v, str):
            if v.lower() == 'none':
                return None
            elif v.lower() in {'nonlinear', 'polynomial'}:
                return v.lower()
            else:
                v = v.replace(' ', '') # get rid of spaces
                result = [i.split(sep2) for i in v.split(sep1)]
                result = np.array(result, dtype=float)
                return result
        else:
            raise argparse.ArgumentTypeError('str value expected.')

    @staticmethod
    def str2dtype(v):
        """
        Helper function for argparser, converts str to torch dtype.

        Inputs:
            v: string
        """
        if isinstance(v, str):
            if v == 'float32':
                return torch.float32
            elif v == 'float64':
                return torch.float64
            elif v == 'int32':
                return torch.int32
            elif v == 'int64':
                return torch.int64
            else:
                raise Exception('Unkown str: {}'.format(v))
        else:
            raise argparse.ArgumentTypeError('str value expected.')

    def list2str(v, sep = '-'):
        """
        Helper function to convert list to string.

        Inputs:
            v: list
        """
        if isinstance(v, list):
            return sep.join([str(i) for i in v])
        else:
            raise Exception('list value expected.')

    @staticmethod
    def float2str(v):
        """
        Helper function to convert float to str in si notation.

        Inputs:
            v: float
        """
        # TODO make this more robust
        if isinstance(v, float):
            vstr = "{:.1e}".format(v)
            if vstr[2] == '0':
                # converts 1.0e-04 to 1e-04
                vstr = vstr[0:1] + vstr[3:]
            if vstr[-2] == '0':
                # converts 1e-04 to 1e-4
                vstr = vstr[0:-2] + vstr[-1]
        else:
            raise Exception('float value expected.')
        return vstr
