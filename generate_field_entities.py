import json
field = "yugioh"
src_f = open("./data/zeshel/documents/{}.json".format(field))
line = src_f.readline()
entitys = []
while line:
    line = json.loads(line)
    if line["title"] in entitys:
        print("error:duplicate entitys")
        break
    entitys.append(line["title"].lower())
    line = src_f.readline()
src_f.close()
print(len(entitys))
with open(field+"_entitys.json",'w') as f:
    f.write(json.dumps(entitys))