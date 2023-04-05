rr = ""
with open("traindata0_origin.txt", "r") as f:
    s = f.read()
    for line in s.splitlines():
        line = line[8:]
        line = line.split(" ")
        print()
        print()
        for l in line:
            if l != "":
                rr = rr + "," + str(l)

print(rr)