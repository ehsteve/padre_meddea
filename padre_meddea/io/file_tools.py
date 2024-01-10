"""
This module provides a generic file reader.
"""
import numpy as np

import ccsdspy
from ccsdspy.utils import split_packet_bytes
from ccsdspy.utils import split_by_apid
from ccsdspy import PacketField, PacketArray

__all__ = ["read_file"]

APID_HIST = 0xA2  # 162
APID_PHOTON = 0xA0  # 160
APID_HK = 0xA3  # 163
APID_CMD = 0xA5  # 165

APID_DICT = {
    APID_HIST: "histogram",
    APID_PHOTON: "photon",
    APID_HK: "housekeeping",
    APID_CMD: "command response",
}


def read_file(data_filename):
    """
    Read a file.

    Parameters
    ----------
    data_filename: str
        A file to read.

    Returns
    -------
    data: str

    Examples
    --------
    """
    result = read_l0_file(data_filename)
    return result


def summarize_file(data_filename):
    """Given a data, provide a summary of the contents."""
    data = parse_l0_file(data_filename)

    for this_apid in data.keys():
        if this_apid in list(APID_DICT.keys()):
            print(f"0x{this_apid:02x} recognized as {APID_DICT[this_apid]} packets.")
            print(f"Found {len(data[this_apid])} packets.")
            number_of_hits = 0
            for this_packet in data[APID_PHOTON]:
                number_of_hits += int(this_packet['PIXEL_DATA'] / 3)
        else:
            print(f"0x{this_apid:02x} unknown")
    
    print(f"Found {len()}")


def print_file(data_filename, num_lines=100, columns=8):
    """Given a binary level 0 file, print out the contents so that it can be visually inspected.

    Parameters
    ----------
    data_filename: str
        A file to read.
    num_lines: int
        The number of lines to print
    columns: int
        The number of columns to use

    Returns
    -------
    None
    """
    inside_packet = False
    packet_start_index = 0
    packet_counter = [0, 0, 0, 0]
    data = np.fromfile(data_filename, dtype=">u2")
    for i, this_num in enumerate(data[0 : columns * num_lines]):
        if (i % columns) == (columns - 1):
            end_char = "\n"
        else:
            end_char = ""

        if not inside_packet:
            if this_num == APID_HIST:  # green
                print(f"\033[92m0x{this_num:04x}\033[00m ", end=end_char)
                packet_length = (data[i + 2] + 1) / 2.0
                packet_start_index = i
                inside_packet = True
                packet_counter[0] += 1
            elif this_num == APID_PHOTON:  # red
                print(f"\033[91m0x{this_num:04x}\033[00m ", end=end_char)
                packet_length = (data[i + 2] + 1) / 2.0
                packet_start_index = i
                inside_packet = True
                packet_counter[1] += 1
            elif this_num == APID_HK:  # yellow
                print(f"\033[93m0x{this_num:04x}\033[00m ", end=end_char)
                packet_counter[2] += 1
            elif this_num == APID_CMD:  # purple
                print(f"\033[95m0x{this_num:04x}\033[00m ", end=end_char)
                packet_counter[3] += 1
        else:
            print(f"\033[47m0x{this_num:04x}\033[00m ", end=end_char)

        if i > (packet_start_index + packet_length):
            inside_packet = False

    print()
    print(f"Read {i+1} 16 bit words ({(i+1)/len(data): 0.2f}% of the file)")
    print(f"Found {packet_counter[0]} Histrogram packets.")
    print(f"Found {packet_counter[1]} photon packets.")
    print(f"Found {packet_counter[2]} hk packets.")
    print(f"Found {packet_counter[3]} command packets.")


def parse_l0_file(data_filename: str, include_ccsds_headers: bool = True):
    """
    Parse a level 0 data file.

    Parameters
    ----------
    data_filename : str
        A file to read.
    include_ccsds_headers : bool
        If True then return the CCSDS headers in the data arrays.

    Returns
    -------
    data : dict
        A dictionary of data arrays. Keys are the APIDs.
    """
    with open(data_filename, "rb") as mixed_file:
        stream_by_apid = split_by_apid(mixed_file)
    result = {}

    if APID_HIST in stream_by_apid.keys():
        packet_def = packet_definition_hist()
        pkt = ccsdspy.FixedLength(packet_def)
        try:
            data = pkt.load(
                stream_by_apid[APID_HIST], include_primary_header=include_ccsds_headers
            )
            result.update({APID_HIST: data})
        except RuntimeError as error:
            print(f"Error parsing {APID_HIST}: {error}")
    if APID_PHOTON in stream_by_apid.keys():
        packet_def = packet_definition_ph()
        pkt = ccsdspy.VariableLength(packet_def)
        try:
            data = pkt.load(
                stream_by_apid[APID_PHOTON], include_primary_header=include_ccsds_headers
            )
            result.update({APID_PHOTON: data})
        except RuntimeError as error:
            print(f"Error parsing {APID_PHOTON}: {error}")

    return result


def read_lvl0_file(data_filename):
    """Read a level 0 data file.

    Args:
        data_filename (_type_): _description_
    """
    data = parse_l0_file(data_filename)
    number_of_hits = 0
    for this_packet in data[APID_PHOTON]:
        number_of_hits += int(this_packet['PIXEL_DATA'] / 3)

    pixel_data = np.zeros(number_of_hits)
    pixel_times = np.zeros(number_of_hits)
    detector_number = np.zeros(number_of_hits)
    pixel_number = np.zeros(number_of_hits)


def packet_definition_hist():
    """Return the packet definition for the histogram packets."""
    # the number of pixels provided by a histogram packet
    NUM_PIXELS = 8 * 4

    p = [
        PacketField(name="START_TIME", data_type="uint", bit_length=4 * 16),
        PacketField(name="END_TIME", data_type="uint", bit_length=4 * 16),
    ]

    for i in range(NUM_PIXELS):
        p += [
            PacketField(name=f"HISTOGRAM_SYNC{i}", data_type="uint", bit_length=8),
            PacketField(name=f"HISTOGRAM_DETNUM{i}", data_type="uint", bit_length=3),
            PacketField(name=f"HISTOGRAM_PIXNUM{i}", data_type="uint", bit_length=5),
            PacketArray(
                name=f"HISTOGRAM_DATA{i}",
                data_type="uint",
                bit_length=16,
                array_shape=512,
            ),
        ]

    p += [PacketField(name="CHECKSUM", data_type="uint", bit_length=16)]

    return p


def packet_definition_ph():
    """Return the packet definition for the photon packets."""
    p = [
        PacketField(name="TIME", data_type="uint", bit_length=4 * 16),
        PacketField(name="INT_TIME", data_type="uint", bit_length=16),
        PacketField(name="FLAGS", data_type="uint", bit_length=16),
        PacketField(name="CHECKSUM", data_type="uint", bit_length=16),
        PacketArray(
            name="PIXEL_DATA", data_type="uint", bit_length=16, array_shape="expand"
        ),
    ]
    return p
