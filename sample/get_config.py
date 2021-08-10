import argparse
import json
import numpy as np

def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--chi', type=str2list, help='chi matrix using latex separator style')
    parser.add_argument('--m', type=int, default=1024, help='number of particles')
    parser.add_argument('--k', type=int, help='number of particle types (inferred if None)')
    parser.add_argument('--save_chi', action="store_true", 'true to save chi to wd')

    args = parser.parse_args()
    return args

def str2list(v, sep1 = '\\', sep2 = '&'):
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
        else:
            v = v.replace(' ', '') # get rid of spaces
            result = [i.split(sep2) for i in v.split(sep1)]
            result = np.array(result, dtype=float)
            return result
    else:
        raise argparse.ArgumentTypeError('str value expected.')


def main():
    with open('default_config.json', 'rb') as f:
        config = json.load(f)

    args = getArgs()

    # save chi to wd
    if args.save_chi:
        np.savetxt('chis.txt', chi, fmt='%0.5f')
        np.save('chis.npy', chi)

    # save chi to config
    rows, cols = args.chi.shape
    letters='ABCDEFG'
    assert rows == cols, "chi not square".format(args.chi)
    for row in range(rows):
        for col in range(cols):
            key = 'chi{}{}'.format(letters[row], letters[col])
            val = args.chi[row, col]
            config[key] = val

    # save nbeads
    config['nbeads']=args.m

    # save chipseq files
    if args.k is None:
        args.k = rows
    else:
        assert args.k == rows, 'number of particle types does not match shape of chi'
    config['chipseq_files'] = ['seq{}.txt'.format(i) for i in range(args.k)]

    print(config)
    with open('config.json', 'w') as f:
        json.dump(config, f, indent = 2)

if __name__ == '__main__':
    main()
