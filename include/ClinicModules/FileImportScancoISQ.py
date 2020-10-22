import numpy as np
import struct


def import_isq(isq_file_path):
    with open(isq_file_path, "rb") as binary_file:
        data_raw = binary_file.read()
    header = read_isq_header(data_raw)
    image_data = np.frombuffer(data_raw[header["data_offset"]:], dtype=np.int16)
    image_data = image_data.reshape((header["dimz_p"], header["dimy_p"], header["dimx_p"]))
    image_data = np.moveaxis(image_data, [0, 1, 2], [2, 1, 0])
    image_data = np.rot90(image_data, 3)
    return (header["dimx_p"], header["dimy_p"], header["dimz_p"]), image_data


def read_isq_header(data_raw):
    # Used some help from here https://github.com/KitwareMedical/ITKIOScanco/blob/master/src/itkScancoImageIO.cxx
    # and from the following Scanco Doc:
    # /* Scanco ISQ Header Information: - note: Scanco uses OpenVMS on Alpha workstations
    #
    # Little endian byte order (the least significant bit occupies the lowest memory position.
    #
    # 00   char    check[16];              // CTDATA-HEADER_V1
    check_passed = data_raw[:16].decode("utf-8") == "CTDATA-HEADER_V1"
    if not check_passed:
        raise NotImplementedError("Could not confirm ISQ data header integrity.")
    # 16   int     data_type;
    data_type = struct.unpack('i', data_raw[16:20])[0]
    # 20   int     nr_of_bytes;
    nr_of_bytes = struct.unpack('i', data_raw[20:24])[0]
    # 24   int     nr_of_blocks;
    nr_of_blocks = struct.unpack('i', data_raw[24:28])[0]
    # 28   int     patient_index;          //p.skip(28);
    patient_index = struct.unpack('i', data_raw[28:32])[0]
    # 32   int     scanner_id;				//p.skip(32);
    scanner_id = struct.unpack('i', data_raw[32:36])[0]
    # 36   int     creation_date[2];		//P.skip(36);
    creation_date = list()
    creation_date.append(struct.unpack('i', data_raw[36:40])[0])
    creation_date.append(struct.unpack('i', data_raw[40:44])[0])
    # 44   int     dimx_p;					//p.skip(44);
    dimx_p = struct.unpack('i', data_raw[44:48])[0]
    # 48   int     dimy_p;
    dimy_p = struct.unpack('i', data_raw[48:52])[0]
    # 52   int     dimz_p;
    dimz_p = struct.unpack('i', data_raw[52:56])[0]
    # 56   int     dimx_um;				//p.skip(56);
    dimx_um = struct.unpack('i', data_raw[56:60])[0]
    # 60   int     dimy_um;
    dimy_um = struct.unpack('i', data_raw[60:64])[0]
    # 64   int     dimz_um;
    dimz_um = struct.unpack('i', data_raw[64:68])[0]
    # 68   int     slice_thickness_um;		//p.skip(68);
    slice_thickness_um = struct.unpack('i', data_raw[68:72])[0]
    # 72   int     slice_increment_um;		//p.skip(72);
    slice_increment_um = struct.unpack('i', data_raw[72:76])[0]
    # 76   int     slice_1_pos_um;
    slice_1_pos_um = struct.unpack('i', data_raw[76:80])[0]
    # 80   int     min_data_value;
    min_data_value = struct.unpack('i', data_raw[80:84])[0]
    # 84   int     max_data_value;
    max_data_value = struct.unpack('i', data_raw[84:88])[0]
    # 88   int     mu_scaling;             //p.skip(88);  /* p(x,y,z)/mu_scaling = value [1/cm]
    mu_scaling = struct.unpack('i', data_raw[88:92])[0]
    # 92	int     nr_of_samples;
    nr_of_samples = struct.unpack('i', data_raw[92:96])[0]
    # 96	int     nr_of_projections;
    nr_of_projections = struct.unpack('i', data_raw[96:100])[0]
    # 100  int     scandist_um;
    scandist_um = struct.unpack('i', data_raw[100:104])[0]
    # 104  int     scanner_type;
    scanner_type = struct.unpack('i', data_raw[104:108])[0]
    # 108  int     sampletime_us;
    sampletime_us = struct.unpack('i', data_raw[108:112])[0]
    # 112  int     index_measurement;
    index_measurement = struct.unpack('i', data_raw[112:116])[0]
    # 116  int     site;                   //coded value
    site = struct.unpack('i', data_raw[116:120])[0]
    # 120  int     reference_line_um;
    reference_line_um = struct.unpack('i', data_raw[120:124])[0]
    # 124  int     recon_alg;              //coded value
    recon_alg = struct.unpack('i', data_raw[124:128])[0]
    # 128  char    name[40]; 		 		//p.skip(128);
    name = data_raw[128:168].decode("utf-8")
    # 168  int     energy;        /* V     //p.skip(168);
    energy = struct.unpack('i', data_raw[168:172])[0]
    # 172  int     intensity;     /* uA    //p.skip(172);
    intensity = struct.unpack('i', data_raw[172:176])[0]
    #
    # ...
    #
    # 508 int     data_offset;     /* in 512-byte-blocks  //p.skip(508);
    data_offset = struct.unpack('i', data_raw[508:512])[0]
    data_offset = (data_offset + 1) * 512
    del data_raw
    return locals()
