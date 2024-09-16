import yaml


def parse_yaml(filename):
    with open(filename) as file:
        try:
            parsed_file = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
        return parsed_file
