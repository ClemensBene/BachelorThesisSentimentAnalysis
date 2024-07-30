import configparser

def read_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    config_dict = {section: dict(config.items(section)) for section in config.sections()}
    config_dict['DEFAULT'] = {key: value for key, value in config.items('DEFAULT')}
    return config_dict