import os
import sys

PATH = "C:\\Users\\gabri\\OneDrive\\Desktop\\BiiMRustCol\\googleImagePull\\urls\\languages\\"
master = open("master.txt", "a+")
for files in os.walk(PATH):
    for filename in files:
        for filed in filename:
            if filed.endswith(".txt"):
                opened = open(PATH + filed, "r")
                writer = opened.read()
                print(writer)
                master.write(writer)
                opened.close()
