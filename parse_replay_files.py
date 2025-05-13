from zephyrus_sc2_parser import parse_replay
from glob import glob
import os
import numpy as np
import time

replay_path = "replays"
metadata_path = "replay_metadata"
abt_file_info_path = "about_file_number.txt"
output_file = "output.csv"

admin_header = [
    "id",
    "file_name",
    "gametime"
]

admin_footer = [
    "winner"
]
# This is the CSV header sans admin. Admin includes player prefixes, ids, etc.
generic_header = [
        "unspent_minerals",
        "unspent_gas",
        "unit_count",
        "building_count",
        "upgrade_count",
        "active_workers",
        "supply_cap",
        "total_gas_collected",
        "total_minerals_collected",
        "total_army_value"
        ]

p1header = ["p1_" + val for val in generic_header]
p2header = ["p2_" + val for val in generic_header]

def get_replay_files():
    replay_files = glob(os.path.join(replay_path, "*.SC2Replay"))
    return replay_files

def parse_replay_files(files):
    replays = list()
    badfilecount = 0
    goodfilecount = 0
    for i, file in enumerate(files):
        try:
            replay = parse_replay(file)
            replay.metadata["filename"] = file
            replays.append(replay)
            goodfilecount += 1
        except:
            os.remove(file)
            badfilecount += 1
        if (i % 1000 == 0):
            print(f"{i}/{len(files)} files parsed...")

    print(f"Parsing Complete:") 
    print(f"\t Replays Failed:    {badfilecount}")
    print(f"\t Replays Succeeded: {goodfilecount}")
    print(f"\tSuccess Rate: {float(goodfilecount) / float(goodfilecount + badfilecount)}")
    return replays

def get_features_from_replay(replay, make_summary_file=False):
    play_features = list()
    for gamestate in replay.timeline:
        return_val = get_features_from_gamestate(gamestate) 
        play_features.append(return_val)

    filename = "nofile"
    if(make_summary_file):
        filename = make_replay_summary_file(replay, play_features)

    for i in range(len(play_features)):
        play_features[i] = [filename] + play_features[i]
    
    return play_features

def make_replay_summary_file(replay, play_features):
    filename = "None"
    num = None
    with open(abt_file_info_path, "r") as f:
        file_cont = f.read()
        num = int(file_cont)
        filename = f"{num:05X}.abt"
        try:
            with open(os.path.join(metadata_path, filename), "w") as summary_file:
                for key, value in replay.metadata.items():
                    summary_file.writelines([str(key), str(value), "\n"])
                for key, value in replay.summary.items():
                    summary_file.writelines([str(key), str(value), "\n"])
                summary_file.write("\n" + ", ".join(generic_header) + "\n")
                for row in play_features:
                    row = [filename] + row
                    summary_file.writelines([str(row), "\n"])
        except:
            # I'm dealing with a lot of errors centered around weird characters in 
            # different languages. Its really rare (<1/10000), so I'll just make these
            # empty.
            filename="nofile"
            print(f"Language string problem found.")
        finally:
            num += 1
            with open(abt_file_info_path, "w") as f:
                f.write(str(num))
            return filename

def get_features_from_gamestate(gamestate): 
    state_features = list([gamestate[1]['gameloop']])
    for player in [1,2]:
        state_features.extend(get_features_for_player_state(gamestate[player]))
    return state_features

def get_features_for_player_state(playerstate):
    resources = playerstate["unspent_resources"]
    unspent_minerals = resources["minerals"]
    unspent_gas = resources["gas"]
    unit_count = len(playerstate["unit"])
    building_count = len(playerstate["building"])
    upgrade_count = len(playerstate["upgrade"])
    active_workers = playerstate["workers_active"]
    supply_cap = playerstate["supply_cap"]
    total_resources_collected = playerstate["resources_collected"]
    total_gas_collected = total_resources_collected["gas"]
    total_minerals_collected = total_resources_collected["minerals"]
    total_army_value = playerstate["total_army_value"]

    # The purpose of all this extra complexity is to make the header dynamic.
    # Dictionary keys are stored in the csvHeader variable, and values are looked
    # up in order for lookup in this dictionary.
    feature_dict = dict({
        "unspent_minerals":unspent_minerals,
        "unspent_gas":unspent_gas,
        "unit_count":unit_count,
        "building_count":building_count,
        "upgrade_count":upgrade_count,
        "active_workers":active_workers,
        "supply_cap":supply_cap,
        "total_gas_collected":total_gas_collected,
        "total_minerals_collected":total_minerals_collected,
        "total_army_value":total_army_value
    })

    feature_list = [feature_dict[header_key] for header_key in generic_header]
    return list(feature_list)




###############################################
# BEGIN MAIN
###############################################

files = get_replay_files()
print(f"""
      Looking for .SC2Replay files
      {len(files)} files found.
      Starting parsing.
      """)

replays = parse_replay_files(files)
total_states = 0
for game in replays:
    total_states += len(game.timeline)

print(f"""
      Parsing Complete.
      Starting Feature Processing on {total_states} observations.
      """)

total_columns = (len(p1header)) + len(p2header) + len(admin_header) + len(admin_footer)

all_features = np.zeros((total_states, total_columns), dtype=object)
row_index = 0
for i,replay in enumerate(replays):  
    obs = get_features_from_replay(replay, make_summary_file=True)
    for observation in obs:
        all_features[row_index] = [row_index] + observation + [replay.metadata["winner"]]
        row_index += 1

np.savetxt(output_file, all_features, "%s", ",", comments="", header=", ".join(admin_header + p1header + p2header + admin_footer))
print(all_features.shape)
print("Done")