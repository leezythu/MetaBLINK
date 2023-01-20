import json
import enchant
import inflect
p = inflect.engine()
d = enchant.Dict("en_US")

def no_overlap(m,title):
    m_l = set(m.split())
    m_l2 = set(p.singular_noun(item) for item in m.split())
    m_l = m_l | m_l2
    t_l = set(title.split())
    if len(m_l & t_l)>0:
        return False
    return True
    
split = "train"
src = open("../data/zeshel/blink_format/{}.jsonl".format(split))
out = open("./data/{}_mentions_all.jsonl".format(split),'w')
line = src.readline()
filtered = 0
cnt = 0
while line:
    line = json.loads(line)
    new_line = {}
    cnt += 1
    #you can decide whether filter out data or not
    if not no_overlap(line["mention"].lower(),line["label_title"].lower()):
        filtered += 1
        line = src.readline()
        continue
    new_line["mention"] = line["mention"]
    new_line["label"] = line["label"]
    # new_line["context_left"] = line["context_left"]
    # new_line["context_right"] = line["context_right"]
    out.write(json.dumps(new_line)+"\n")
    line = src.readline()
print("filtered:",filtered)
print("cnt:",cnt)