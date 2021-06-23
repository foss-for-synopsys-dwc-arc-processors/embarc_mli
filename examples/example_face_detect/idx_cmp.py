import numpy as np
import os
import sys
import glob
import idx2numpy


def print_difference(np_ref, np_tst, is_csv_out=False):
    if not is_csv_out:
        print("DIFFERENCE:")
    np_dif = np_ref - np_tst
    eps = 0.0000000000001
    num_diff_elements = np.count_nonzero(np_dif)
    total_elements = len(np_dif)
    prcnt_diff_elements = num_diff_elements/total_elements * 100
    snr = 10.0 * np.log10((np.sum(np_ref.astype(np.float)**2) + eps) / (np.sum(np_dif.astype(np.float)**2) + eps))
    if is_csv_out:
        print("{},{},{:.2f},{},{},{:.2f}".format(total_elements, num_diff_elements, prcnt_diff_elements, np_dif.min(), np_dif.max(), snr))
    else:
        print("\tElements are differs: {} of {} ({:.2f} %)".format(num_diff_elements, total_elements, prcnt_diff_elements))
        print("\tMin/Max diff: [{}, {}]".format(np_dif.min(), np_dif.max()))
        print("\tSNR: {:.2f} dB".format(snr))
    

if __name__ == '__main__':
    #np_ref = idx2numpy.convert_from_file("c:\MyData\Sources\ML_IOT\\2020_05_29_32BitScales_experiments\em7d_ssd_3_mli_latest\\28_CONV_2D.idx").reshape((10, 10, 256))
    #np_tst = idx2numpy.convert_from_file("c:\MyData\Sources\ML_IOT\\2020_05_29_32BitScales_experiments\em7d_ssd_4_mli_ref_32bit\\28_CONV_2D.idx").reshape((10, 10, 256))
    #np_dif = np_ref - np_tst
    #np_dif_T = np.transpose(np_dif, (2, 0, 1))
    #np_ref = idx2numpy.convert_from_file(
    #    "c:\MyData\Sources\ML_IOT\\2020_05_29_32BitScales_experiments\em7d_ssd_1_tflm_baseline\\7_CONV_2D.idx").reshape(
    #    (75, 75, 128))
    #np_tst = idx2numpy.convert_from_file(
    #    "c:\MyData\Sources\ML_IOT\\2020_05_29_32BitScales_experiments\em7d_ssd_4_mli_ref_32bit\\7_CONV_2D.idx").reshape(
    #    (75, 75, 128))
    #np_dif = np_ref - np_tst
    is_csv_out = False
    if len(sys.argv) > 3 and sys.argv[3].lower() == 'csv':
        print("File Name,Total Elements,Diff Elements,Diff Elements%,Min Diff,Max Diff,SNR")
        is_csv_out = True
    else:
        print(sys.argv)
    cmp_dir_1 = sys.argv[1]
    cmp_dir_2 = sys.argv[2]

    files_to_cmp = [os.path.basename(f_name) for f_name in glob.glob(cmp_dir_1 + "/*.idx") if os.path.isfile(f_name)]
    files_to_cmp_1 = [os.path.join(cmp_dir_1, f_name) for f_name in files_to_cmp]
    files_to_cmp_2 = []
    for f_name in files_to_cmp:
        file_path = os.path.join(cmp_dir_2, f_name)
        files_to_cmp_2.append(file_path if os.path.exists(file_path) else None)
    for file_name, file_ref, file_tst in zip(files_to_cmp, files_to_cmp_1, files_to_cmp_2,):
        print(file_name, end=": " if not is_csv_out else ",")
        if file_tst is None:
            print("NO APPROPRIATE IDX FILE IN TEST!")
            continue
        np_ref = idx2numpy.convert_from_file(file_ref).flatten()
        np_tst = idx2numpy.convert_from_file(file_tst).flatten()
        if np.array_equal(np_ref, np_tst):
            if is_csv_out:
                print("{},{},{:.2f},{},{},{:.2f}".format(len(np_ref), 0, 0.0, 0, 0, 0.0))
            else:
                print("Bitwise!")
        elif np_ref.size != np_tst.size:
            print("SHAPE MISMATCH!")
        else:
            print_difference(np_ref, np_tst, is_csv_out)

