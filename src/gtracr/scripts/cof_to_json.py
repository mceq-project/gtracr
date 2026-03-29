"""
Converts the .COF file, which contains the spherical harmonic coefficients of the magnetic potential in the IGRF model, into a /JSON format. This is done since:
- the file format can be read universally (both in Unix and Windows)
- it is human readable
- lots of support exists with this file format

This should be run every time there is an update with the .COF files obtained from the IGRF website.
"""

import json
import os

IGRF_VERSION = 13  # should be updated with each IGRF version

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PARENT_DIR, "data")
COF_PATH = os.path.join(DATA_DIR, f"IGRF{IGRF_VERSION}.COF")
JSON_PATH = os.path.join(DATA_DIR, f"igrf{IGRF_VERSION}.json")


def export_json(igrf_dict):
    """
    Export the file as a json format
    """
    with open(JSON_PATH, "w") as f:
        json.dump(igrf_dict, f, indent=2, separators=(",", ":"))


def add_coefficients(igrf_dict, model_dict):
    """
    Add the coefficients to the dictionary igrf_dict.
    they are evaluated in C++ where we have something like:
    gh = [g10, h10, g11, h11, g20, h20, g21, h21, ...]
    but hn0 = 0, so these are effectively removed, resulting in:
    gh = [g10, g11, h11, g20, g21, h21, ...]

    same thing applies to secular variation coefficients
    """

    with open(COF_PATH) as f:
        lines = [line.rstrip() for line in f]  # get each line as a string
        for line in lines:
            line_elems = str.split(line)  # get contents of each line

            if len(line_elems) == 8:  # coefficient line
                # contents are given in the following order:
                # n, m, g, h, g_dot, h_dot, model, line_num

                (n, m, g, h, gdot, hdot, model, line_num) = line_elems

                # cast values to int / float
                n = int(n)
                m = int(m)
                g = float(g)
                h = float(h)
                gdot = float(gdot)
                hdot = float(hdot)

                # find date specific to each model
                date = model_dict[model]

                # append coefficients and secular variation
                igrf_dict[date]["gh"].append(g)
                igrf_dict[date]["gh_sv"].append(gdot)
                if m != 0:  # only append h when m != 0
                    igrf_dict[date]["gh"].append(h)
                    igrf_dict[date]["gh_sv"].append(hdot)

    # return igrf_dict


def make_igrfdict():
    """
    Read the .COF file and store the data into different data structures
    and store them in a model dictionary
    """
    # initialize arrays to store content
    model_arr = []
    epoch_arr = []
    nmain_arr = []  # max1 in original
    nsv_arr = []  # max2 in original
    nac_arr = []  # max2 in original
    yrmin_arr = []
    yrmax_arr = []
    altmin_arr = []
    altmax_arr = []

    # read the file
    with open(COF_PATH) as f:
        lines = [line.rstrip() for line in f]  # get each line as a string
        for line in lines:
            line_elems = str.split(line)  # get contents of each line
            if len(line_elems) != 8:  # model line
                #             print(line_elems)
                # truncate
                line_elems = line_elems[:-2]
                #             print(line_elems)
                # unpack
                (model, epoch, nmain, nsv, nac, yrmin, yrmax, altmin, altmax) = (
                    line_elems
                )

                # append to array
                model_arr.append(model)
                epoch_arr.append(epoch + "0000")  # needed for C++ reading .json files
                nmain_arr.append(int(nmain))
                nsv_arr.append(int(nsv))
                nac_arr.append(int(nac))
                yrmin_arr.append(float(yrmin))
                yrmax_arr.append(float(yrmax))
                altmin_arr.append(float(altmin))
                altmax_arr.append(float(altmax))

            else:  # coefficient line
                continue  # pass for now, do this for each model

    # create the framework for the json file
    igrf_dict = dict((epoch, {}) for epoch in epoch_arr)
    # print(igrf_dict)
    for i, date in enumerate(list(igrf_dict.keys())):
        igrf_dict[date]["model"] = model_arr[i]
        igrf_dict[date]["nmain"] = nmain_arr[i]
        igrf_dict[date]["nsv"] = nsv_arr[i]
        igrf_dict[date]["nac"] = nac_arr[i]
        igrf_dict[date]["yrmin"] = yrmin_arr[i]
        igrf_dict[date]["yrmax"] = yrmax_arr[i]
        igrf_dict[date]["altmin"] = yrmin_arr[i]
        igrf_dict[date]["altmax"] = altmax_arr[i]

        igrf_dict[date]["gh"] = []
        igrf_dict[date]["gh_sv"] = []
    # print(igrf_dict)

    # create way to find specific date based on model
    model_dict = dict((model_arr[i], epoch_arr[i]) for i in range(len(model_arr)))

    return igrf_dict, model_dict


def cof_to_json():
    """
    Convert the .COF file into .JSON format
    """

    # read the file and contain the model data
    igrf_dict, model_dict = make_igrfdict()

    # add the coefficients to the igrf_dict
    add_coefficients(igrf_dict, model_dict)

    # export resulting dictionary as .json file
    export_json(igrf_dict)


if __name__ == "__main__":
    cof_to_json()
