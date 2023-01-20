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
    
def contains_number(m):
    for item in m.split():
        if item.isdigit():
            return True
    return False

def check(mention):
    l = mention.split()
    for item in l:
        if not d.check(item):
            return False
    return True

field = "yugioh"
succ = 0
cnt = 0
src_f = open("../exact_{}.jsonl".format(field))
assi_f = open("./data/rewrited_{}.jsonl".format(field))

out_f = open("../data/zeshel/blink_format/filtered_{}.jsonl",'w')
src_line = src_f.readline()
assi_line = assi_f.readline()

exist_entitys = []
exist_mentions = []

while src_line and assi_line:
    src_line = json.loads(src_line)
    assi_line = json.loads(assi_line)
    cnt+=1
    title = src_line["label_title"].lower()
    new_m = assi_line["predict"]
    new_m = new_m.replace("<pad>","").replace("</s>","").strip().lower()
    if len(new_m.split()) > 2 and len(new_m.split()) < 4 and title not in exist_entitys and new_m not in exist_mentions and no_overlap(new_m,title) and not contains_number(new_m) and check(new_m):
        succ+=1
        src_line["mention"] = new_m
        exist_entitys.append(title)
        exist_mentions.append(new_m)
        out_f.write(json.dumps(src_line)+"\n")
    src_line = src_f.readline()
    assi_line = assi_f.readline()
