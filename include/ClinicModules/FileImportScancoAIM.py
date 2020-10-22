#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import struct
import io


# ---------------------AIM HEADER--------------------------------
# Pricipial Adresses in the AIM Header from SCANCO.
# All Adresses are Bytes and from a Matlab Programm. NOT VALIDATED!

# There are AIM Version 2 and 3 -> This Script is at the Moment only capable
# to work with AIMv2 and uncompressed UINT16 Datatype.
# Distinction AIMv2 and AIMv3:
# 0 7 = 'AIMDATA' -> V3 and currently not supported!

# HEADER AIMv2

# 0  20 preheader all uint32
#   0  4  length preheader (should be 20!)
#   4  8  length aimheader
#   8  12 length prolog
#   12 16 length data
#   16 20 additional data
# headeroffset =  length preheader + length aimheader + length prolog

# 20 40 head - uint32/VAXD
# 40 44 aimtype - specifies datatype - uint32,see import_aim function
# 44 56 position - unit32
# 56 68 dimension - uint32
#   56 60 X
#   60 64 Y
#   64 68 Z

# HEADER AIMv3 -> See Matlabfiles in Jira


# file_path -> path to AIM file as string
# returns numpyarray:
# dimX, dimY, dimZ, Data
def import_aim(file_path) -> (tuple, np.array):
    with open(file_path, "rb") as binary_file:
        data_raw = binary_file.read()
    header_offset, dim, datatype_aim, length_data = read_header_aim(data_raw)
    dim = dim.astype(int)

    # calculate and compare datatype magicstrings
    # return ->  magicstring -> name
    # 1*2^16+1  -> 8_bit integer
    # 2*2^16+2  -> 16_bit_integer
    # 3*2^16+4  -> 32_bit_integer
    # 8*2^16+2  -> DT compresses
    # 26*2^16+4 -> 64_bit_float
    # 21*2^16+1 -> 8_bit_binary_compressed
    # at the moment only 16 bit integer is supported!

    if datatype_aim == (1 * np.power(2, 16) + 1):
        assert False, "AIM DATATYPE is 8bit Integer -> not supported at the Moment."
    elif datatype_aim == (2 * np.power(2, 16) + 2):
        # Cutout only the imagedata
        image_data_byte = data_raw[header_offset:header_offset + length_data]
        image_data_array = np.frombuffer(image_data_byte, dtype=np.int16).reshape((dim[2], dim[1], dim[0]))
        image_data_array = np.moveaxis(image_data_array, [0, 1, 2], [2, 1, 0])
    elif datatype_aim == (3 * np.power(2, 16) + 4):
        assert False, "AIM DATATYPE is 32bit Integer -> not supported at the Moment."
    elif datatype_aim == (8 * np.power(2, 16) + 2):
        assert False, "AIM is DT Compressed -> not supported at the Moment."
    elif datatype_aim == (26 * np.power(2, 16) + 4):
        assert False, "AIM DATATYPE is 64bit float -> not supported at the Moment."
    elif datatype_aim == (21 * np.power(2, 16) + 1):
        # Header info
        """ 1           - header_offset 
            2, 3, 4     - dim' 
            5, 6, 7     - position' 
            8, 9, 10    - el_size_mm' 
            11          - aim_type 
            12          - length_data 
            13          - AimVer 

        dat = fread(fid,header_info(12),'uint8=>uint8');
        dat(end+1)=0; % add one element due to loop construction
        
        if (header_info(13)==2) %AimVer
            val1 = dat(5);
            val2 = dat(6);
            field_offs = 7;
        else %AimVer 3
            val1 = dat(9);
            val2 = dat(10);
            field_offs = 11;
        end
        """
        image_data_byte = data_raw[header_offset:header_offset + length_data]
        val1 = image_data_byte[5 - 1]
        val2 = image_data_byte[6 - 1]
        field_offs = 7
        """
        cur_len = dat(field_offs);
        
        if (cur_len == 255)
            cur_len = 254;
            change_val = false;
        else
            change_val = true;
        end
        
        cur_val = val1;
        is_value_1 = true;
        """
        cur_len = image_data_byte[field_offs - 1]
        if cur_len == 255:
            cur_len = 254
            change_val = False
        else:
            change_val = True
        cur_val = val1
        is_value_1 = True
        """
        vol=zeros(header_info(2),header_info(3),header_info(4),'uint8');
        """
        vol = np.zeros((dim[0], dim[1], dim[2]), dtype=np.uint8)
        """
        for k=1:header_info(4)
            for j=1:header_info(3)
                for i=1:header_info(2)
                    vol(i,j,k) = cur_val;
                    cur_len = cur_len - 1;
                    if (cur_len == 0)
                        if (change_val)
                            is_value_1 = ~is_value_1;
                            if (is_value_1)
                                cur_val = val1;
                            else
                                cur_val = val2;
                            end
                        end
                        field_offs = field_offs + 1;
                        cur_len = dat(field_offs);
                        if (cur_len==255)
                            cur_len=254;
                            change_val = false;
                        else
                            change_val = true;
                        end
        """
        for k in range(dim[2]):
            for j in range(dim[1]):
                for i in range(dim[0]):
                    vol[i, j, k] = cur_val
                    cur_len = cur_len - 1
                    if cur_len == 0:
                        if change_val:
                            is_value_1 = not is_value_1
                            if is_value_1:
                                cur_val = val1
                            else:
                                cur_val = val2
                        field_offs = field_offs + 1
                        if field_offs >= len(image_data_byte):
                            break
                        else:
                            cur_len = image_data_byte[field_offs - 1]
                        if cur_len == 255:
                            cur_len = 254
                            change_val = False
                        else:
                            change_val = True
        """
                    end
                end
            end
        end
        
        vol2 = typecast(vol(:),'int8');
        vol = reshape(vol2,size(vol));
        """
        image_data_array = vol.astype(np.int8)
    else:
        assert False, "Datatype not supported!"
    return dim, image_data_array


# rawdata from open() operation
# returns header_offset as bytes, dimensions(X,Y,Z),Datatype, length data
def read_header_aim(data_raw):
    # Check AIM Version
    # fixme Implement check

    # Calculate Headeroffset

    # prehead
    data = data_raw[0:4]  # as uint32
    length_prehead = struct.unpack('I', data)[0]
    assert length_prehead == 20, "headerlength prehead has to be 20. Correct filetype?"

    # aimhead
    data = data_raw[4:8]
    length_aimhead = struct.unpack('I', data)[0]

    # prolog
    data = data_raw[8:12]
    length_prolog = struct.unpack('I', data)[0]

    # data
    data = data_raw[12:16]
    length_data = struct.unpack('I', data)[0]

    header_offset = length_prehead + length_aimhead + length_prolog

    # get datatype magicstring
    data = data_raw[40:44]
    datatype_aim = struct.unpack('I', data)[0]

    # get dimensions
    data = data_raw[56:68]
    dim_data = struct.iter_unpack('I', data)
    dim = np.zeros(3)  # XYZ
    iter_Counter = 0
    for i in dim_data:
        dim[iter_Counter] = i[0]
        iter_Counter = iter_Counter + 1

    return header_offset, dim, datatype_aim, length_data


def write_header_aim(file: io.BufferedWriter, info: dict):
    # Prehead
    length_prehead = 20
    file.write(struct.pack("I", length_prehead))

    length_aimhead = 140
    file.write(struct.pack("I", length_aimhead))

    length_prolog = 20
    file.write(struct.pack("I", length_prolog))

    raw_data_offset = length_prehead + length_aimhead + length_prolog

    length_data = info["length_data"]
    file.write(struct.pack("I", length_data))

    add_data = 0
    file.write(struct.pack("I", add_data))

    # Head
    file.write(struct.pack("IIIII", 0, 0, 0, 0, 0))

    datatype_aim = int(2 * np.power(2, 16) + 2)
    file.write(struct.pack("I", datatype_aim))

    position = info["position"]  # Tuple of int
    file.write(struct.pack("III", *position))

    dimensions = info["dim"]  # Tuple of int
    file.write(struct.pack("III", *dimensions))

    offset = info["offset"]  # Tuple of int
    file.write(struct.pack("III", *offset))

    supdim = info["supdim"]  # Tuple of int
    file.write(struct.pack("III", *supdim))

    suppos = info["suppos"]  # Tuple of int
    file.write(struct.pack("III", *suppos))

    subdim = info["subdim"]  # Tuple of int
    file.write(struct.pack("III", *subdim))

    testoff = info["testoff"]  # Tuple of int
    file.write(struct.pack("III", *testoff))

    el_size_mm = info["el_size_mm"]  # Tuple of int
    file.write(struct.pack("III", *el_size_mm))

    return raw_data_offset

