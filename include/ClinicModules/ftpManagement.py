import os


def download_files_from_ct(ftp_object, patient_number, measurement_number, dicom_number, file_ending, storagefolder):
    # test both drives since we don't know if the Files are already in the archive

    try:
        ftp_object.cwd('DK0' + build_path_download(patient_number, measurement_number))
    except:
        try:
            ftp_object.cwd('DISK4' + build_path_download(patient_number, measurement_number))
        except:
            return 1

    try:
        files = ftp_object.nlst()
    except:
        return 2
    for file in files:
        if (fileName := build_file_name(dicom_number, file_ending)) in file:
            base, extension = os.path.splitext(file)
            fileName += ";" + extension.split(";")[1]

            try:
                if os.path.exists(os.path.join(storagefolder + fileName)):
                    print(f"File {os.path.join(storagefolder, fileName)} already exists.")
                    return 0
                ftp_object.retrbinary('RETR ' + fileName, open(storagefolder + fileName, 'wb').write)

            except:
                # If in DK0 and DK4 the folders exist, but only on DISK4 the right data.
                try:
                    ftp_object.cwd('DISK4' + build_path_download(patient_number, measurement_number))
                    ftp_object.retrbinary('RETR ' + fileName, open(storagefolder + fileName, 'wb').write)
                except:
                    return 2
    return 0


def build_path_download(patient_number, measurement_number):
    if len(patient_number) != 8 | len(measurement_number) != 8:
        return 0
    path_download = ':[MICROCT.DATA.' + patient_number + '.' + measurement_number + ']'
    return path_download


def build_file_name(dicom_number, extension):
    if len(dicom_number) != 7:
        return 0
    file_name = 'C' + dicom_number + extension
    return file_name
