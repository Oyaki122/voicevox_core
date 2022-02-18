from ctypes import *
import platform
import os
from pathlib import Path

import numpy

int64_dim1_type = numpy.ctypeslib.ndpointer(dtype=numpy.int64, ndim=1)
float32_dim1_type = numpy.ctypeslib.ndpointer(dtype=numpy.float32, ndim=1)

int64_dim2_type = numpy.ctypeslib.ndpointer(dtype=numpy.int64, ndim=2)
float32_dim2_type = numpy.ctypeslib.ndpointer(dtype=numpy.float32, ndim=2)

get_os = platform.system()
get_arch = platform.machine()

lib_file = ""
if get_os == "Windows":
    lib_file = "core.dll"
elif get_os == "Darwin":
    lib_file = "libcore.dylib"
elif get_os == "Linux":
    lib_file = "libcore.so"

lib = cdll.LoadLibrary(
    str(Path(os.path.dirname(__file__) + f"/../../core/lib/{lib_file}")))

lib.initialize.argtypes = (c_char_p, c_bool, c_int)
lib.initialize.restype = c_bool

lib.finalize.argtypes = ()

lib.metas.restype = c_char_p

lib.supported_devices.restype = c_char_p

lib.yukarin_s_forward.argtypes = (
    c_int64, int64_dim1_type, int64_dim1_type, float32_dim1_type)
lib.yukarin_s_forward.restype = c_bool

lib.yukarin_sa_forward.argtypes = (c_int64, int64_dim2_type, int64_dim2_type, int64_dim2_type,
                                   int64_dim2_type, int64_dim2_type, int64_dim2_type, int64_dim1_type, float32_dim2_type)
lib.yukarin_sa_forward.restype = c_bool

lib.decode_forward.argtypes = (
    c_int64, c_int64, float32_dim2_type, float32_dim2_type, int64_dim1_type, float32_dim1_type)
lib.decode_forward.restype = c_bool

lib.last_error_message.restype = c_char_p


def initialize(root_dir_path: str, use_gpu: bool, cpu_num_threads: int):
    success = lib.initialize(root_dir_path.encode(), use_gpu, cpu_num_threads)
    if not success:
        raise Exception(lib.last_error_message().decode())


def metas() -> str:
    return lib.metas().decode()


def supported_devices() -> str:
    return lib.supported_devices().decode()


def yukarin_s_forward(length: int, phoneme_list: numpy.ndarray, speaker_id: numpy.ndarray) -> numpy.ndarray:
    output = numpy.zeros((length, ), dtype=numpy.float32)
    success = lib.yukarin_s_forward(length, phoneme_list, speaker_id, output)
    if not success:
        raise Exception(lib.last_error_message().decode())
    return output


def yukarin_sa_forward(
    length: int,
    vowel_phoneme_list,
    consonant_phoneme_list,
    start_accent_list,
    end_accent_list,
    start_accent_phrase_list,
    end_accent_phrase_list,
    speaker_id
):
    output = numpy.empty((len(speaker_id), length,), dtype=numpy.float32)
    success = lib.yukarin_sa_forward(
        length, vowel_phoneme_list, consonant_phoneme_list, start_accent_list, end_accent_list, start_accent_phrase_list, end_accent_phrase_list, speaker_id, output
    )
    if not success:
        raise Exception(lib.last_error_message().decode())
    return output


def decode_forward(length: int, phoneme_size: int, f0, phoneme, speaker_id):
    output = numpy.empty((length*256,), dtype=numpy.float32)
    success = lib.decode_forward(
        length, phoneme_size, f0, phoneme, speaker_id, output
    )
    if not success:
        raise Exception(lib.last_error_message().decode())
    return output


def finalize():
    lib.finalize()


if __name__ == "__main__":
    initialize("./release", False, 8)
    print(metas())
    finalize()
