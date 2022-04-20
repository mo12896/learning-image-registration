import json



class Params(object):
    def __init__(self, config_path):
        self.path = config_path
        self.__dict__ = None
        self.parse_json_file()

    def parse_json_file(self):
        file = open(self.path)
        # Deserializing JSON string into a python class object
        self.__dict__ = json.loads(file)

