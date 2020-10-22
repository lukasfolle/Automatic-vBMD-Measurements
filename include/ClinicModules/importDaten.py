#!/usr/bin/python
# -*- coding: latin-1 -*-
# Import of Patientdata from CSV file.
# Filelayout:
# XCTPatientNumber;MeasurmentNumber;DicomNumber;CCPStatus(0/1 - 99 if error)

def importPatientData(FileName):
    with open(FileName, 'r') as patientDataFile:
        patientData = patientDataFile.readlines()

    # Cutoff lineendings

    for x in range(len(patientData)):
        patientData[x] = patientData[x].split('\n')[0]  # windows
        patientData[x] = patientData[x].split('\r')[0]  # mac

    # Split the lines in seperate arrays
    patNumber = list(range(len(patientData)))
    messNumber = list(range(len(patientData)))
    dicomNumber = list(range(len(patientData)))

    for y in range(len(patientData)):

        patNumber[y], messNumber[y], dicomNumber[y] = patientData[y].split(';')[:3]
        # zfill padds numbers to appropriate length (constrained by
        # folderstructure on the XCT)
        patNumber[y] = patNumber[y].zfill(8)
        messNumber[y] = messNumber[y].zfill(8)
        dicomNumber[y] = dicomNumber[y].zfill(7)
    return patNumber, messNumber, dicomNumber
