import stanza
import json,re
# # stanza.download('en')       # This downloads the English models for the neural pipeline
nlp = stanza.Pipeline('en') # This sets up a default neural pipeline in English
field = "yugioh"

f = open(field+"_entitys.json")
field_entitys = json.load(f)
f.close()

max_num = 5000

def exist(text,entitys):
    flag = False
    label_title = None
    match_type = None
    if text in entitys:#exact match
        flag = True
        label_title = text
        match_type = "exact match"
        return flag,label_title,match_type
    return flag,label_title,match_type


src = open("./data/zeshel/documents/"+field+".json")
out_f = open("exact_"+field+"_tmp.jsonl",'w')
cnt = 0
succ_cnt = 0
line = src.readline()
finish_flag = False
while line:
    if finish_flag:
        break
    cnt += 1
    if cnt%100==0:
        print(cnt)
    text = json.loads(line)["text"]
    from_doc = json.loads(line)["title"]
    try:
        doc = nlp(text)
    except:
        line = src.readline()
        continue
    mention_list = []
    entity_list = []
    doc_flag = False
    for sentence in doc.sentences:
        if doc_flag:
            break
        for ent in sentence.ents:
            if doc_flag:
                break
            if ent.text in mention_list:
                continue
            if ent.type == "DATE" or ent.type == "CARDINAL" or ent.type == "ORDINAL":#不要日期
                continue
            flag, label_title,match_type = exist(ent.text.lower(),field_entitys)
            if flag:
                if from_doc.lower() == label_title:
                    continue
                start,end = ent.start_char,ent.end_char
                sample = {}
                sample["context_left"] = text[:start]
                if len(sample["context_left"].split())<64:
                    continue
                sample["context_left"] = " ".join(sample["context_left"].split()[-128:])
                sample["mention"] = ent.text 
                sample["context_right"] = text[end:]
                sample["context_right"] = " ".join(sample["context_right"].split()[:128])
                sample["label_title"] = label_title
                sample["world"] = field
                sample["from_doc"] = from_doc
                sample["match_type"] = match_type
                out_f.write(json.dumps(sample)+"\n")
                succ_cnt += 1
                if succ_cnt>=max_num:
                    finish_flag = True
                entity_list.append(label_title)
                mention_list.append(ent.text)
                doc_flag = True
    line = src.readline()
src.close()
out_f.close()

f = open(field+"_entity_index_by_title")
entity_index_by_title = json.load(f)
f.close()

out_f = open("exact_"+field+".jsonl",'w')
f = open("exact_"+field+"_tmp.jsonl")
line = f.readline()
while line:
    line = json.loads(line)
    new_line = {}
    new_line["context_left"] = line["context_left"]
    new_line["context_right"] = line["context_right"]
    label_title = line["label_title"]
    new_line["label"] = entity_index_by_title[label_title][0]
    new_line["label_title"] = label_title
    new_line["label_id"] = entity_index_by_title[label_title][1]
    new_line["mention"] = line["mention"]
    new_line["from_doc"] = line["from_doc"]
    new_line["world"] = line["world"]
    new_line["match_type"] = line["match_type"]
    out_f.write(json.dumps(new_line)+"\n")
    line = f.readline()
f.close()
out_f.close()