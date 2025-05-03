import requests
import numpy as np
import random
import math
import tarfile
import io

header = """# date=2025_01_30_19_01_47
# version=2.4.27
# algorithmVersion=1.16.7
# gc={"size":{"w":220,"h":220}}
# gc={"offset":{"x":0,"y":0}}
# gc={"start":{"x":0,"y":0.0000}}
# gc={"keys":["x","y"],"rm":1}
# timeConfig=
G90
G0 F3000
G4M1
M9064 B2
M9039 C2
# GS002 HEAD
G0 F180000
M4 S0
G1 F180000
G0 X0 Y0










G102
G91
#G0Z4F600
G103
G90
G0 F180000
"""

footer = """
# END
# GS002 TAIL



G102
#G91
#G0Z-4F600
G103




G90
G0 S0
G0 F180000
G1 F180000
M536 U0
M6 P1

"""

# x0,y0,x1,y1 by N
N = 100
the_lines = np.random.random(size=(N, 4))*50.0

def make_cut_gcode(paths,
        Z = 20.0,   # mm
        power = 100.0, # %
        speed = 10000.0, #mm/s  
    ):
    # G0 move XY
    # G1 cut, XY s=power, f=feedrate
    # G0Q30 frequency 30-60

    # Put your gcode here
    
    contents = f"""
    # Do this when the laser changes:
    # GS002 VECTOR HEAD
    # motion_start
    G4M1
    #G21=Blue or G22=fiber for which laser to use
    G21
    G90
    G0Q30

    G4M1
    M523P40

    # Z move
    G102
    #G91 #incremental
    G0Z{Z}F600
    #G90 #absolute
    G103
    G0F180000

    """
    def round3(v):
        return round(v, 3) 

    parts = []

    for fooPath in paths:
        x0, y0 = fooPath[0]
        parts.append(f"G0X{round3(x0)}Y{round3(y0)}")
        
        for x, y in fooPath[1:]:
            parts.append(f"G1X{round3(x)}Y{round3(y)}S{power*10.0}F{speed*60.0}")

    parts.append("")

    contents += '\n'.join(parts)

    # // TODO: Return z to start
    # result.push("#".to_string());
    # result.push(format!("G0Z{}", round3(23.0)));

    return contents

def make_xf(contents):
    filename = "F1 Ultra-template.xf"
    t = tarfile.open(filename, 'r')
    files = t.getmembers()

    tar_fileobj = io.BytesIO()   
    
    #output = tarfile.open('the_test_file.xf','w')
    output = tarfile.open(fileobj=tar_fileobj, mode='w')
    
    # tarfile.TarInfo("preview.jpg")
    # tarfile.TarInfo("motion.gcode")
    # tarfile.TarInfo("description.json")
    # tarfile.TarInfo("border.gcode")

    for part in files:
        f = t.extractfile(part)
        data = f.read()
        print(part.name, data[:100])

        if part.name == 'motion.gcode':
            info = tarfile.TarInfo("motion.gcode")
            data = header+contents+footer
            data = data.replace('\n', '\r\n')
            info.size = len(data)
            output.addfile(info, io.BytesIO(data.encode()))
            #print(data.encode())
        else:
            info = tarfile.TarInfo(part.name)
            info.size = len(data)
            output.addfile(info, io.BytesIO(data))
    output.close()
    print("Done")
    tar_fileobj.seek(0)
    return tar_fileobj.read()

xf_data = make_xf(make_cut_gcode(the_lines))


# The url to use
base_url = "http://192.168.1.239"
# Get the camera image (a jpg) with the same settings that xtool uses
data = requests.get(f"{base_url}:8329/camera/snap?width=4656&height=3496&timeOut=30000")

# The jpg data
print(data.content)
# Save it to a file
with open("camera_apture.jpg", "wb") as f:
    f.write(data.content)
    f.close()

data = '{"action":"goTo","z":20.0,"stopFirst":1,"F":5000}'
requests.put(f"{base_url}:8080/focus/control", data=data)

"""
POST /processing/upload?gcodeType=processing&fileType=xf&taskId=PC_F1Ultra_MXFK002B2024072307949AB_1740275842761&autoStart=0 HTTP/1.1
Accept: application/json, text/plain, */*
Content-Type: application/octet-stream
Content-Length: 7680
User-Agent: axios/1.7.7
Accept-Encoding: gzip, compress, deflate, br
Host: 192.168.1.233:8080
Connection: close
"""

the_file_contents = xf_data # open("the_test_file.xf", "rb").read()
ok = requests.post(f"{base_url}:8080/processing/upload?gcodeType=processing&fileType=xf&taskId=PC_F1Ultra_MXFK002B2024072307949AB_1740275842761&autoStart=1", data=the_file_contents)
print("ok? = ", ok)
