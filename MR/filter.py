
field = "yugioh"
succ = 0
cnt = 0
src_f = open("../exact_"+field+".jsonl")
assi_f = open("./data/inference_"+field+"_res_w_mlm.jsonl")

out_f = open("../data/zeshel/blink_format/sync_mg_"+field+"_mlm.jsonl",'w')
src_line = src_f.readline()
assi_line = assi_f.readline()

exist_entitys = []
exist_mentions = []

while src_line and assi_line:
    src_line = json.loads(src_line)
    assi_line = json.loads(assi_line)
    if src_line["mention"].lower() == assi_line["golden_label"].lower():
        cnt+=1
        title = src_line["label_title"].lower()
        new_m = assi_line["predict"]
        new_m = new_m.replace("<pad>","").replace("</s>","").strip().lower()
        # if len(new_m.split())>2 and len(new_m.split())<4 and title not in exist_entitys and new_m not in exist_mentions and no_overlap(new_m,title) and not contains_number(new_m) and check(new_m):
        # if check(new_m):
        if True:
            succ+=1
            src_line["mention"] = new_m
            exist_entitys.append(title)
            exist_mentions.append(new_m)
            out_f.write(json.dumps(src_line)+"\n")
    src_line = src_f.readline()
    assi_line = assi_f.readline()
