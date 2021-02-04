import math

def Distance(x1,y1,x2,y2):
    d = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return d
def Connected(link1, link2):
    validCon = False
    seperation = Distance(link1[0],link2[0],link1[1],link2[1])
    r1 = link1[2]
    r2 = link2[2]
    if r1 + r2 + .25 > seperation:
        validCon = True
    return validCon

input1 = [[10, 20, 12], [20, 1, 12], [40,30,12], [50,20,12],[40,1,12],[20,30,12],[0,0,0],[25,10,5],[10,10,12],[10,25,5],[25,25,12],[25,25,12],[0,0,0],[-1,-1,-1]]
necklace = []
validNeck = False
# ask if there will be at least 3 links in each necklace?
# are the links in order?
for i in range(0, len(input1)-1):
    necklaceChecked = 1
    if input1[i] != [0,0,0]:
        necklace.append(input1[i])
    else:
        #algorithm
        # the index of the arrays inside this matrix refers to a link
        # inside each array is a list of the connections the link has.
        connectionMatrix = []
        for j in range(0, len(necklace)):
            ringJ = []
            for k in range(0, len(necklace)):
                ringJ.append(Connected(necklace[j],necklace[k]))
            connectionMatrix.append(ringJ)
        
        #wipe neclase and output
        if validNeck:
            valOInval = "valid"
        else:
            valOInval = "invalid"
        print(f"Case {necklaceChecked}: {valOInval}")
        necklace = []



