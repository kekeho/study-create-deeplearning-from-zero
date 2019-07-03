import pickle
import json
import requests

pickle_obj_url = 'https://github.com/oreilly-japan/deep-learning-from-scratch/raw/master/ch03/sample_weight.pkl'

# Download pickle object
pickle_bin = requests.get(pickle_obj_url).content
network = pickle.loads(pickle_bin)


new = dict()
for k, v in network.items():
    if k[0] == 'b':  # bias
        new[k] = list(map(float, v))
    else:
        new[k] = [[float(x) for x in l] for l in v]

with open('sample_weight.json', 'w') as f:
    json.dump(new, f)
