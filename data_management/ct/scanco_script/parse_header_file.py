import os
import pandas as pd

from arthritis_utils.General import path_exists_check


def remove_unnecessary_characters(string):
    string = string.replace("!>", "").replace("!", "").replace("-" * 79, "")
    return string


def create_dict_from_header_lines(header_lines):
    info = dict()
    for line in header_lines:
        if len(line) > 0:
            info_line = ",".join(line.split())
            line_pieces = info_line.split(",")
            for i, piece in enumerate(line_pieces):
                try:
                    converted = float(piece)
                except ValueError:
                    converted = piece
                line_pieces[i] = converted
            if line_pieces[0] in info.keys():
                line_pieces[0] += "_1"
            info[line_pieces[0]] = (line_pieces[1:])
    return info


def parse_header(file_path: str) -> dict:
    path_exists_check(file_path)
    with open(file_path, "r") as f:
        header = f.read()

    header = remove_unnecessary_characters(header)
    header_lines = header.split("\n")
    info = create_dict_from_header_lines(header_lines)

    return info


def find_latest_gobj_version(row, df):
    pd.options.mode.chained_assignment = None
    same_file_names = df[df["file_name"] == row["file_name"]]
    same_file_names = same_file_names.sort_values("version", ascending=False)
    latest_version = same_file_names["version"].iloc[0]
    return latest_version


def filter_header(df):
    print(f"INFO: Found {len(df['file_name'].unique())} unique patient files.")
    pd.options.mode.chained_assignment = None
    df_filtered = df[(150 < df["dim_x"]) & (df["dim_x"] < 280) &
                     (150 < df["dim_y"]) & (df["dim_y"] < 280) &
                     (50 < df["dim_z"]) & (df["dim_z"] < 250)]

    df_filtered["latest_version"] = df_filtered.apply(find_latest_gobj_version, args=(df_filtered,), axis=1)
    df_filtered = df_filtered[df_filtered["version"] == df_filtered["latest_version"]]

    print(f"INFO: After filtering found {len(df_filtered['file_name'].unique())} patient files.")
    return df_filtered


def get_df_from_header_folder(header_folder):
    db = dict()
    num_dropped = 0
    for file in os.listdir(header_folder):
        header_path = os.path.join(header_folder, file)
        header_file = parse_header(header_path)
        if "Original" not in header_file.keys():
            num_dropped += 1
            continue
        gobj_name = header_file["Original"][1]
        db[gobj_name] = header_file

    print(f"INFO: Dropped {num_dropped} files since gobjs were empty.")

    df = pd.DataFrame.from_dict(db, orient="index")
    df["file_name"] = df.apply(lambda row: row["Original"][1].split(".")[0], axis=1)
    df["version"] = df.apply(lambda row: int(row["Original"][1].split(";")[1]), axis=1)
    df["dim_x"] = df.apply(lambda row: int(row["dim"][0]), axis=1)
    df["dim_y"] = df.apply(lambda row: int(row["dim"][1]), axis=1)
    df["dim_z"] = df.apply(lambda row: int(row["dim"][2]), axis=1)
    df["pos_x"] = df.apply(lambda row: int(row["pos"][0]), axis=1)
    df["pos_y"] = df.apply(lambda row: int(row["pos"][1]), axis=1)
    df["pos_z"] = df.apply(lambda row: int(row["pos"][2]), axis=1)
    df = df.set_index(pd.Index(range(len(df))))
    df_filtered = filter_header(df)
    return df_filtered
