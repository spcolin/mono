"""
for compute avg eval
"""

f=open("/home/colin/Documents/rdlossrst3/without4.txt")

absrel=0
silog=0
log10=0
rms=0
squa=0
logrms=0

line=f.readline()
while line:
    if "absREL" in line:
        data=line.split(':')[-1].replace('\n','')
        absrel=absrel+float(data)
    elif "silog" in line:
        data=line.split(':')[-1].replace('\n','')
        silog=silog+float(data)
    elif "log10" in line:
        data=line.split(':')[-1].replace('\n','')
        log10=log10+float(data)
    elif "RMS" in line:
        data=line.split(':')[-1].replace('\n','')
        rms=rms+float(data)
    elif "squaRel" in line:
        data=line.split(':')[-1].replace('\n','')
        squa=squa+float(data)
    elif "logRms" in line:
        data=line.split(':')[-1].replace('\n','')
        logrms=logrms+float(data)

    line=f.readline()

f.close()

f2=open("/home/colin/Documents/rdlossrst3/without4_avg.txt","w")

# print("absREL:",absrel/10)
# print("silog:",silog/10)
# print("log10:",log10/10)
# print("RMS:",rms/10)
# print("squaRel:",squa/10)
# print("logRms:",logrms/10)
f2.write("absREL:"+str(absrel/10)+"\n")
f2.write("silog:"+str(silog/10)+"\n")
f2.write("log10:"+str(log10/10)+"\n")
f2.write("RMS:"+str(rms/10)+"\n")
f2.write("squaRel:"+str(squa/10)+"\n")
f2.write("logRms:"+str(logrms/10)+"\n")


f2.close()
