import os
from include.ClinicModules import importDaten as ip
from include.ClinicModules import ftpManagement as fp
from ftplib import FTP
import tqdm

# ----------------- Configuration Area --------------------------

version = 'V2.0 - only AIM (int16 uncompressed) PSA and RA'
storageFolderTemp = os.path.join("C:", "Temp")

# ----------------- / Configuration Area --------------------------

print('---------------- Export form imagedata to external drive --------------------------')
print('started by clinic.py in version: ' + version)
print('Temporary Storage: ' + storageFolderTemp)
file = r"C:\Zwischenspeicher\2020-08-28 MIIEEB-234 Export MCP Nur MG3.CSV"
print('start import CSV Data from file: ' + file)
xct, meas, dicom = ip.importPatientData(file)

print('Import of patientdata finished. We imported  '
      + str(len(xct)) + 'rows.')

print('User checked that CT was unoccupied and no measurement can be interrupted ')

print('Initiating Connection to CT... : ')
ftp = FTP('ip_adress')
pw = input('enter password for microCT')
ftp.login('user_name', pw)
print('FTP Connection established.')

print('Download of Files:')
dicomFound = list()
CCPFound = list()
file_endings = list()

for x in tqdm.tqdm(range(len(dicom))):
    file_endings_to_export = ["_SEG.AIM"]
    for file_ending_to_export in file_endings_to_export:
        return_value = fp.download_files_from_ct(ftp, xct[x], meas[x], dicom[x], file_ending_to_export,
                                                 storageFolderTemp)

        if return_value == 0:
            dicomFound.append(dicom[x])
            # CCPFound.append(ccp_status[x])
            file_endings.append(file_ending_to_export)
        elif return_value == 1:
            print('ERROR:  Folder for Patientnumber ' + str(xct[x]) + ' and Measurementnumber '
                  + str(meas[x]) + ' not found!')
        else:
            print('ERROR:  Not all Files for Patientnumber ' + str(xct[x]) + ', Measurementnumber '
                  + str(meas[x]) + ' and DICOM  ' + str(dicom[x]) + ' found!')

print("Success.")
for dicom, ccp, ending in zip(dicomFound, CCPFound, file_endings):
    print(dicom, ccp, ending)
