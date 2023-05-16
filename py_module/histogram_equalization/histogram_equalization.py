import histogram_equalization._histogram_equalization as he
import numpy as np
import warnings

EqType = {
    "mono": 0,
    "bi": 1
}

ImgFmt = {
    "RGB": 0,
    "YCbCr": 1
}

def __get_eq_alg_id( eq_alg: str ):
    if eq_alg not in EqType:
        raise ValueError( "Unknown equalization algorithm '{0}'. Allowed algorithms: {1}"
                          .format( eq_alg , list(EqType.keys()) ) )
    return EqType[eq_alg]

def __get_img_fmt_id( img_fmt: str ):
    if img_fmt not in ImgFmt:
        raise ValueError( "Unsupporterd image format '{0}'. Supported formats: {1}"
                          .format( img_fmt , list(ImgFmt.keys()) ) )
    return ImgFmt[img_fmt]

def __check_params( window_size: int, n_values: int ):
    assert window_size >= 3 , "window_size must be >= 3"
    assert n_values >= 1, "n_values must be >= 1"

def __check_image( img: np.ndarray ):
    if len(img.shape) != 3:
        raise ValueError( "Expected a 3 dimensional numpy array, but img.shape={0}."
                          .format( img.shape ) )
    if img.shape[2] != 3:
        raise ValueError( "Expected a 3-channel image with shape (W,H,3), but img.shape={0}"
                          .format( img.shape ) )
    if img.dtype != np.ubyte:
        warnings.warn( "Input image dtype ({0}) is not np.ubyte. The image will be casted to np.ubyte."
                       .format(img.dtype) )
        return img.astype( np.ubyte )
    return img

def hist_equalization( img: np.ndarray , window_size: int, n_values: int=256, eq_alg: str="mono" , img_fmt: str="RGB"):
    __check_params(window_size , n_values )
    img = __check_image( img )
    eq_type = __get_eq_alg_id( eq_alg )
    fmt = __get_img_fmt_id(img_fmt)
    return he.hist_equalization( img , window_size , n_values , eq_type , fmt )

def hist_equalization_omp( img: np.ndarray , window_size: int, n_values: int=256, eq_alg: str="mono", img_fmt: str="RGB" , n_threads: int=4):
    __check_params(window_size, n_values)
    img = __check_image(img)
    eq_type = __get_eq_alg_id(eq_alg)
    fmt = __get_img_fmt_id(img_fmt)
    return he.hist_equalization_omp( img , window_size , n_values , eq_type , fmt , n_threads )

def hist_equalization_cuda( img: np.ndarray , window_size: int, eq_alg: str="mono" , img_fmt: str="RGB" , pbw: int=64, pbh: int=64):
    __check_params(window_size, 256)
    img = __check_image(img)
    eq_type = __get_eq_alg_id(eq_alg)
    fmt = __get_img_fmt_id(img_fmt)
    return he.hist_equalization_cuda( img , window_size , eq_type , fmt , pbw , pbh )

def init_cuda_context():
    he.init_cuda_context()