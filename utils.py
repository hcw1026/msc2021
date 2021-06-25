import yaml

def parse_config(yaml_path):
    """parse_config: convert a yaml path into the corresponding dictionary
    Input:
        - yaml_path: str
            - the path of a yaml config
    """
    loader = yaml.SafeLoader
    
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=loader)

    return config

def get_linear_layer_variables(module):
    """obtain all linear layer variables in a snt.Module
    Input:
        - module: snt.Module
            - the module to get the tf.Variable from
    Output:
        - list of tf.Variables which are the linear weights of a MLP and not bias
    """
    variables = []
    for var in module.trainable_variables:
        layer, weight_bias = var.name.split("/")[-2:]
        if layer.startswith("linear") and weight_bias[0]=="w":
            variables.append(var)
    return variables
