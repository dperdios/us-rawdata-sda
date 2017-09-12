import os
import re
from typing import List


def check_trained_networks(path: str) -> List:
    # Check valid network directories
    valid_dir = [f for f in os.listdir(path=path) if os.path.isdir(os.path.join(path, f)) and f.startswith('SDA-')]
    valid_dir.sort()

    # Configuration of the network (in order to restore it)
    #   Regular expressions
    reg_cp = re.compile(r'(?<=-CP)\d*(?=-)')
    reg_ln = re.compile(r'(?<=-LN)\d*(?=-)')
    reg_rf = re.compile(r'(?<=-RF)\d*')

    valid_configs = []
    for nm in valid_dir:
        # Initialize dict
        net_cfg = dict.fromkeys(['compression_percent', 'layer_number', 'learn_mm', 'redundancy_factor'])
        # Compression learned or not
        if 'CNL' in nm:
            net_cfg['learn_mm'] = False
        elif 'CL' in nm:
            net_cfg['learn_mm'] = True
        else:
            raise NameError('Invalid dir name')
        # Compression percent, layer number and redundancy factor
        net_cfg['compression_percent'] = int(reg_cp.findall(nm)[0])  # Technically only a single match
        net_cfg['layer_number'] = int(reg_ln.findall(nm)[0])  # Technically only a single match
        net_cfg['redundancy_factor'] = int(reg_rf.findall(nm)[0])  # Technically only a single match
        valid_configs.append(net_cfg)

    return valid_configs
