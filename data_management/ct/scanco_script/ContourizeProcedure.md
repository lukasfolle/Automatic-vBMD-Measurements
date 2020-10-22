## Get slices of interest (z-dimension)
1. Create small annotations in first and last slice of interest for given patient
2. Using ipl `write_part header` write header to file

## Call model
1. Specify image_paths, header_paths, and mezSeite (left or right hand) when calling inference.py on workstation
Output is at same location as image_paths with suffix "_PRED.AIM"

## Convert prediction to contour
1. ipl `read` prediction with suffix _PRED.AIM
2. ipl `convert_to_type` to type char
3. ipl `togobj_from_aim` to get contour file

## Evaluation
Run EVAL Script using the created GOBJ, CHECK contour before running EVAL
