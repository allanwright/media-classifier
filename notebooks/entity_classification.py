#%%
import numpy as np
import pandas as pd
import re

df = pd.read_csv('data/interim/stacked.csv')

#%%
def split_season_episode(name):
    match = re.search(r'(?P<sid>s\d+)(?P<eid>e\d+)', name)
    if match != None:
        name = name.replace(
            match.group(0),
            match.group('sid') + ' ' + match.group('eid'))
    return name

print(split_season_episode('Game.of.Thrones.s01e01.mp4'))
print(split_season_episode('TEST'))

#%%
