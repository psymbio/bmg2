import os
import json
parameters = json.load(open(os.path.join(os.path.dirname(__file__), "parameters.json")))
n_elements = 118
alloy_max_len = 20