#!/usr/bin/python
# -*- coding: latin-1 -*-

# Write text in colsole and/or in Logfile

def logWrite(fileName, text, commandline):
    import time
    try:
        logFile = open(fileName, 'a')
    except:
        logFile = open(fileName, 'w')

    logFile.writelines(time.strftime("%d/%m/%Y") + ' ' + time.strftime("%H:%M:%S") + ': ' + text + '\n')
    if commandline:
        print(text + '\n')
    logFile.close()
    pass
