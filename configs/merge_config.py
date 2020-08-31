
import pdb
def merge_config(config, parser):
    config_key_set = config.keys()
    for key in parser.__dict__.keys():
        if key not in config_key_set:
            raise KeyError("key: {:} is not registered in config file".format(key))
        else:
            if parser.__dict__[key]:
                config[key] = parser.__dict__[key]

    return config