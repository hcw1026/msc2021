import yaml
import re

def parse_config(yaml_path):
    '''parse_config: convert a yaml path into the corresponding dictionary
    Input:
        - yaml_path: str
            - the path of a yaml config
    '''
    loader = yaml.SafeLoader
    
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=loader)

    return config