#!/usr/bin/env python3

import szlfile

szl = szlfile.SzlFile("Onera.szplt")


print(f"file type: {szl.type}")

print(f"dataset title {szl.title}")
print(f"num vars {szl.num_vars}")
print(f"num zones {szl.num_zones}")
print(f"aux items count {szl.num_auxdata_items}")

for i in range(szl.num_zones):
    print(f"zone {i+1} type {szl.zones[i].type}")
    print(f"zone {i+1} title {szl.zones[i].title}")
    print(f"zone {i+1} is enabled: {szl.zones[i].is_enabled()}")
    print(f"zone {i+1} solution time: {szl.zones[i].solution_time}")
    print(f"zone {i+1} strand id: {szl.zones[i].strand_id}")

    for j in range(szl.num_vars):
        print(f"zone {i+1} var {j+1} name: {szl.zones[i].variables[j].name}")
        print(f"zone {i+1} var {j+1} type: {szl.zones[i].variables[j].type}")
        print(f"zone {i+1} var {j+1} is enabled: {szl.zones[i].variables[j].is_enabled()}")
        print(f"zone {i+1} var {j+1} location: {szl.zones[i].variables[j].value_location}")
        print(f"zone {i+1} var {j+1} is passive: {szl.zones[i].variables[j].is_passive()}")
        print(f"zone {i+1} var {j+1} shared zone: {szl.zones[i].variables[j].shared_zone}")
        print(f"zone {i+1} var {j+1} num values: {szl.zones[i].variables[j].num_values}")
        print(f"zone {i+1} var {j+1} values: {szl.zones[i].variables[j].values}")
        print(f"zone {i+1} node map: {szl.zones[i].node_map}")
