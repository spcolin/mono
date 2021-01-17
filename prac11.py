import os



path="/home/colin/Documents/rdlossrst3/withoutavg/"
file_list=os.listdir(path)


def best():
    absREL = 10.0
    silog = 10.0
    log10 = 10.0
    RMS = 10.0
    squaRel = 10.0
    logRms = 10.0

    for i in file_list:
        if i.startswith("with"):
            file_path = path + i
            print(file_path)
            f=open(file_path)

            line=f.readline()
            while line:
                if "absREL" in line:
                    data = line.split(':')[-1].replace('\n', '')
                    if absREL>float(data):
                        absREL= float(data)
                elif "silog" in line:
                    data = line.split(':')[-1].replace('\n', '')
                    if silog>float(data):
                        silog =  float(data)
                elif "log10" in line:
                    data = line.split(':')[-1].replace('\n', '')
                    if  log10 > float(data):
                        log10 = float(data)
                elif "RMS" in line:
                    data = line.split(':')[-1].replace('\n', '')
                    if RMS > float(data):
                        RMS= float(data)
                elif "squaRel" in line:
                    data = line.split(':')[-1].replace('\n', '')
                    if squaRel > float(data):
                        squaRel= float(data)
                elif "logRms" in line:
                    data = line.split(':')[-1].replace('\n', '')
                    if logRms > float(data):
                        logRms= float(data)

                line = f.readline()

            f.close()


    f2=open(path+"bestall.txt","w")

    f2.write("absREL:"+str(absREL)+"\n")
    f2.write("silog:"+str(silog)+"\n")
    f2.write("log10:"+str(log10)+"\n")
    f2.write("RMS:"+str(RMS)+"\n")
    f2.write("squaRel:"+str(squaRel)+"\n")
    f2.write("logRms:"+str(logRms)+"\n")


    f2.close()



def avg():
    absREL = 0
    silog = 0
    log10 = 0
    RMS = 0
    squaRel = 0
    logRms =0

    for i in file_list:
        if i.startswith("with"):
            file_path = path + i
            print(file_path)
            f=open(file_path)

            line=f.readline()
            while line:
                if "absREL" in line:
                    data = line.split(':')[-1].replace('\n', '')
                    absREL= absREL + float(data)
                elif "silog" in line:
                    data = line.split(':')[-1].replace('\n', '')

                    silog =  silog +float(data)
                elif "log10" in line:
                    data = line.split(':')[-1].replace('\n', '')
                    log10 = log10+ float(data)
                elif "RMS" in line:
                    data = line.split(':')[-1].replace('\n', '')
                    RMS= RMS+float(data)
                elif "squaRel" in line:
                    data = line.split(':')[-1].replace('\n', '')
                    squaRel= squaRel+float(data)
                elif "logRms" in line:
                    data = line.split(':')[-1].replace('\n', '')
                    logRms= logRms + float(data)

                line = f.readline()

            f.close()


    f2=open(path+"avgall.txt","w")

    f2.write("absREL:"+str(absREL/len(file_list))+"\n")
    f2.write("silog:"+str(silog/len(file_list))+"\n")
    f2.write("log10:"+str(log10/len(file_list))+"\n")
    f2.write("RMS:"+str(RMS/len(file_list))+"\n")
    f2.write("squaRel:"+str(squaRel/len(file_list))+"\n")
    f2.write("logRms:"+str(logRms/len(file_list))+"\n")


    f2.close()

best()
