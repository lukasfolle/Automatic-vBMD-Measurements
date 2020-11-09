import os


def gobj_to_aim(gobj_filename: str, output_var: str, peel_iter: str):
    return f"/gobj_to_aim {gobj_filename} {output_var} {peel_iter}"


def write_part(variable: str, type: str, file_output: str):
    return f"/write_part {variable} {file_output} {type} binary"


def write(variable: str, file_output: str, compress_type: str, version_020: str):
    return f"/write {variable} {file_output} {compress_type} {version_020}"


def combine(commands: list):
    combined_commands = ""
    for command in commands:
        combined_commands += command + "\n"
    return combined_commands


if __name__ == "__main__":
    gobj_path = r"Fill_in_path"
    gobj_files = [os.path.join(gobj_path, gobj_name) for gobj_name in os.listdir(gobj_path)]
    commands = ""
    scanco_folder = "DK0:[MICROCT.TEST"
    # After calling collect_gobis, they have to be transfered back to the CT using FileZilla any folder.
    # IPL script have to be executed in this scanco folder afterwards.
    # After that, copy header and seg.aim back to workstation.
    for gobj_file in gobj_files:
        commands += "ipl\n"
        gobj_name = os.path.basename(gobj_file)
        gobj_aim_command = gobj_to_aim(gobj_name, "temp", "0")
        base_name = os.path.splitext(gobj_name)[0]
        version = gobj_name.split(";")[1]
        write_part_command = write_part("temp", "header", scanco_folder + "HEADER]" + base_name + "_HEAD.TXT;" + version)
        write_command = write("temp", scanco_folder + "AIM]" + base_name + "_SEG.AIM;" + version, "bin", "true")
        commands += (combine([gobj_aim_command, write_part_command, write_command])) + "/delete temp\n"
        commands += "quit\n\n"
    print(commands)
    with open("GOBJ_TO_AIM.COM", "w") as file:
        file.write(commands)
