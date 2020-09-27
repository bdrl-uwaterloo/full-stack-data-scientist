patn = []
cur = -1

f = open("patent_data.dat", "r")
for line in f:
    if line.startswith("PATN"):
        cur_dic = {
            "APN":None,
            "TTL":None,
            "INVENTOR":None}
        patn.append(cur_dic)
        cur += 1
    if line.startswith("APN"):
        value = line[5:-1]
        patn[cur]["APN"] = value
    if line.startswith("TTL"):
        value = line[5:-1]
        patn[cur]["TTL"] = value
    if line.startswith("INVT"):
        temp = f.readline()
        value = temp[5:-1].split(";")
        patn[cur]["INVENTOR"] = value
f.close()
print(patn)