import os
import numpy as np
from numpy.linalg import pinv, matrix_rank
import nibabel as nib
from os.path import join as opj
from glob import glob

import scipy as sp
from scipy.stats import pearsonr, binom, linregress


def apply_mask(betas, mask):
    """
    Applies mask on input beta values. It assigns zero to those voxels that are NaN but inside the mask.

    Args:
        betas: numpy array containing beta images for all conditions. shape: `(conditions, nbvoxels)`
        mask: 1D numpy array of size `nbvoxels`

    Returns:
        Masked betas as a numpy array.
    """

    nbvox_allowed = np.sum(mask == 1)

    masked = np.zeros((betas.shape[0], nbvox_allowed), dtype='float32')
    for row in range(betas.shape[0]):
        beta = betas[row]
        temp = beta[mask == 1]

        nan_pos = np.isnan(temp)
        nan_cnt = nan_pos.sum()
        if nan_cnt > 0 and row == 0:
            print(f"#NaNs inside the mask: {nan_cnt}")
        temp[nan_pos] = 0

        assert (temp.shape[0] == nbvox_allowed)

        masked[row] = temp

    return masked


def prepare_beta(subject, betas, mask):
    """
    Masks brain representation vectors (it puts zero for those voxels that are NaN but inside the mask) and
    computes the inverse covariance matrix which will be used for decoding.

    Args:
        subject: subject's name
        betas: numpy array containing beta images for all conditions. shape: `(conditions, nbvoxels)`
        mask: 1D numpy array of size `nbvoxels`
    """

    # masking
    reps = apply_mask(betas, mask)

    # inverse cov
    p = np.matmul(reps, reps.T)
    icov = pinv(p)
    rank = matrix_rank(p)
    if rank < icov.shape[0]:
        print(f'RANK DEFICIENT covariance matrix!!! rank is {rank} instead of {icov.shape[0]}')

    return reps, icov, rank


def load_masks_intersect(mask_addresses):
    """
    Loads and returns the intersection of a series of masks. Masks will be flattened.

    Args:
        mask_addresses: list of mask addressess
    """
    for i, mp in enumerate(mask_addresses):
        mask_img = nib.load(mp)
        if i == 0:
            mask = mask_img.get_fdata().reshape((-1))
        else:
            mask = mask * mask_img.get_fdata().reshape((-1))
    return mask


def load_masks_union(mask_addresses):
    """
    Loads and returns the union of a series of masks. Masks will be flattened.

    Args:
        mask_addresses: list of mask addressess
    """
    for i, mp in enumerate(mask_addresses):
        mask_img = nib.load(mp)
        if i == 0:
            mask = mask_img.get_fdata().reshape((-1))
        else:
            mask = mask + mask_img.get_fdata().reshape((-1))
    return np.sign(mask)


def load_betas(beta_addresses, test_set=False):
    """
    Loads and returns a numpy array of shape `(len(beta_addresses), nbvoxels)`.

    Args:
        beta_addresses: list of beta addresses
        test_set: if it is `True`, the function will return `image_ids` as well
    """
    betas = len(beta_addresses) * [None]
    if test_set:
        image_ids = len(beta_addresses) * [None]
    for row, add in enumerate(beta_addresses):
        print(add)
        beta_img = nib.load(add)

        if test_set:
            beta_desc = str(beta_img.header['descrip'])
            image_ids[row] = (beta_desc[beta_desc.find('Sn(1) ') + 6:beta_desc.find('*bf(1)')])

        beta_data = beta_img.get_fdata().reshape((-1))
        betas[row] = beta_data

    if test_set:
        return np.array(betas), np.array(image_ids)
    return np.array(betas)


def fix_test_reps(test_betas):
    """
    Subtract average test representation from all test representations.

    Args:
        test_betas: 2D numpy array containing all beta values for test images. shape: `(nb_samples, nbvoxels)`

    Returns:
        2D numpy array contatining fixed betass
    """
    avg_beta = test_betas.mean(axis=0)
    return test_betas - avg_beta


def get_explicit_masks_info(subjects, explicit_mask_folder):
    """
    Returns explicit mask info.
    """
    mask_list = ['LH_HVC',
                 'LH_V2v',
                 'LH_LOC',
                 'RH_V2v',
                 'RH_hV4',
                 'LH_FFA',
                 'RH_HVC',
                 'RH_V3d',
                 'RH_LOC',
                 'RH_V1v',
                 'LH_V3d',
                 'LH_PPA',
                 'LH_V1d',
                 'RH_PPA',
                 'RH_FFA',
                 'RH_V1d',
                 'RH_V2d',
                 'LH_V3v',
                 'LH_hV4',
                 'LH_V2d',
                 'RH_V3v',
                 'LH_V1v']

    mask_list_both_h = list(set([m[3:] for m in mask_list]))

    masks_info = dict()  # names and addresses
    for subject in subjects:
        masks_info[subject] = dict()
        # key: mask name
        # val: mask address(s)

        # implicit mask (intersection of spm implicit test and train masks)
        masks_info[subject]['spm'] = [opj(explicit_mask_folder, f"{subject}_full_mask.nii")]

        # combine lh and rh masks by authors
        for mask in mask_list_both_h:
            masks_info[subject][mask] = [
                opj(explicit_mask_folder, f"_mask_name_LH_{mask}_subject_id_{subject}", 'Coregister',
                    f"ra{subject}_mask_LH_{mask}.nii"),
                opj(explicit_mask_folder, f"_mask_name_RH_{mask}_subject_id_{subject}", 'Coregister',
                    f"ra{subject}_mask_RH_{mask}.nii")]
    return masks_info


def main_prepare_mapping(subjects, betas_folder, explicit_mask_folder, model_name, latent_size, include_bias,
                         out_folder):
    """
    For each mask, masks the beta vectors and computes the corresponding inverse covariance matrix.
    The results will be saved as numpy files. Each reps. and inverse covariance matrix will be saved
    as a single file called `[subject]_[mask_name]_[model_name]_decode_map` or
    `[subject]_[mask_name]_[model_name]_b_decode_map` if the bias term is included.

    Args:
        subjects: a sequence of subject ids
        betas_folder: address of the beta folder generated by GLM with parametric modulators (latent betas).
        explicit_mask_folder: address of the explicit masks provided by Kamitani
        model_name: name of the model whose latent space is in use
        latent_size: size of the latent vectors
        include_bias: if the `stim` beta vector (bias term) should be included
        out_folder: address of the folder where the files will be saved
    """

    # Loading betas
    betas_lat_info = dict()  # addresses
    for subject in subjects:
        betas_lat_info[subject] = [opj(betas_folder, subject, f'beta_{s:04d}.nii') for s in range(4, latent_size + 4)]

        if include_bias:
            # we add bias term (stimulus regressor) at the end
            betas_lat_info[subject].append(opj(betas_folder, subject, f'beta_{3:04d}.nii'))

    masks_info = get_explicit_masks_info(subjects, explicit_mask_folder)

    for subject in subjects:
        print(subject)
        betas_lat = load_betas(betas_lat_info[subject])
        for mask_name in masks_info[subject]:
            print(mask_name)
            if mask_name == 'spm':
                mask = load_masks_intersect(masks_info[subject][mask_name])
            else:
                mask = load_masks_union(masks_info[subject][mask_name])
            reps, icov, rank = prepare_beta(subject, betas_lat, mask)

            if include_bias:
                out_file = opj(out_folder, subject, f"{subject}_{mask_name}_{model_name}_b_decode_map")
            else:
                out_file = opj(out_folder, subject, f"{subject}_{mask_name}_{model_name}_decode_map")
            np.savez(out_file, betas=reps, invcov=icov, covbetarank=rank, mask=mask)


def load_prepared_beta(path):
    """
    Loads a saved prepared beta file.

    Args:
        path: prepared beta file path

    Returns:
        betas, inverse covariance matrix, rank, preparation mask
    """
    decode_beta = np.load(path)
    return decode_beta['betas'], decode_beta['invcov'], decode_beta['covbetarank'], decode_beta['mask']


def predict_latent(subject, test_patterns, betas, inverse_cov):
    """
    Predicts latent vectors for test images given their corresponding beta images.

    Args:
        test_patterns: numpy array containing beta values for each test image. shape: (nb_test_image, nbvoxels)
        betas: beta values computed for latent features based on training samples. shape: (latent_size, nbvoxels)
        inverse_cov: inverse covariance matrix of `betas`
        image_ids: numpy array of test image ids
        train_beta_path: path to the corresponding decode_beta file
        out_folder: folder address to save the results

    Returns:
        numpy array containing predicted latent vector of each test image. shape: (nb_test_image, latent_size)
    """
    predictions = np.matmul(test_patterns, betas.T)
    predictions = np.matmul(predictions, inverse_cov)

    return predictions


def main_prediction(subjects, betas_folder_test, nb_test, map_folder, model_name, explicit_mask_folder, include_bias,
                    results_folder):
    for subject in subjects:
        print(subject)
        beta_path = [opj(betas_folder_test, subject, f'beta_{s:04d}.nii') for s in range(3, nb_test + 3)]
        patterns, image_ids = load_betas(beta_path, test_set=True)
        patterns_fixed = fix_test_reps(patterns)

        masks_info = get_explicit_masks_info(subjects, explicit_mask_folder)

        for mask_name in masks_info[subject]:
            print(mask_name)
            if include_bias:
                decode_beta_path = opj(map_folder, subject, f"{subject}_{mask_name}_{model_name}_b_decode_map.npz")
            else:
                decode_beta_path = opj(map_folder, subject, f"{subject}_{mask_name}_{model_name}_decode_map.npz")
            betas_lat, invcov, _, mask = load_prepared_beta(decode_beta_path)

            patterns_masked = apply_mask(patterns_fixed, mask)
            predictions = predict_latent(subject, patterns_masked, betas_lat, invcov)
            if include_bias:
                predictions = predictions[:, :-1]

            out_file = opj(results_folder, subject, f"{subject}_{mask_name}_{model_name}_decode_pred")
            np.savez(out_file, stim_ids=image_ids, predictions=predictions)


#################
# Input Arguments
#################
import argparse

parser = argparse.ArgumentParser(
    description="Predicts latent vectors for test images in two stages. (1) compute inverse mapping. (2) predict latent vectors.")
parser.add_argument('betas_folder', type=str,
                    help='address of the folder containing beta files for the GLM with modulation on training sessions. It should be the parent folder of subjects')
parser.add_argument('betas_folder_test', type=str,
                    help='address of the folder containing beta files for the GLM  on testing sessions. It should be the parent folder of subjects')
parser.add_argument('explicit_mask_folder', type=str,
                    help='address of the folder containing explicit masks by Kamitani lab')
parser.add_argument('model_name', type=str,
                    help='name of the model whose latent vectors are going to be used. This name will appear in the results folder name')
parser.add_argument('latent_size', type=int,
                    help='size of the latent vectors (number of dimensions of the latent space)')
parser.add_argument('nb_test', type=int, help='number of test images')
parser.add_argument('map_folder', type=str,
                    help='address of the folder to save (and load) output files for mapping stage (stage 1). the folder will be created if does not exist')
parser.add_argument('prediction_folder', type=str,
                    help='address of the folder to save output files for prediction stage (stage 2). the folder will be created if does not exist')
parser.add_argument('-s', '--subjects', metavar='N', type=str, nargs='+',
                    help='sequence of subjects in the form of sub-01 sub-02 ...',
                    default=['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05'])
parser.add_argument('-b', '--bias', dest='include_bias', action='store_const',
                    const=True, default=False,
                    help='includes stimulus regressor in computations (bias term)')
parser.add_argument('-m', '--map', dest='compute_map', action='store_const',
                    const=True, default=False,
                    help='performs stage (1). If the inverse mappings are already available, you can skip this step by not setting this flag')

args = parser.parse_args()

betas_folder = args.betas_folder
betas_folder_test = args.betas_folder_test
explicit_mask_folder = args.explicit_mask_folder
model_name = args.model_name
latent_size = args.latent_size
nb_test = args.nb_test
map_folder = args.map_folder
prediction_folder = args.prediction_folder
subject_list = args.subjects
include_bias = args.include_bias
compute_map = args.compute_map

def create_dir(root, subjects):
    if not os.path.exists(root):
        print("Creating output directory root ...")
        os.mkdir(root)
    for sub in subjects:
        print(f"Creating output directory for subject {sub} ...")
        if not os.path.exists(opj(root, sub)):
            os.mkdir(opj(root, sub))

def main():
    if compute_map:
        #if not os.path.exists(map_folder):
         #   print("Creating output directory for stage 1 ...")
          #  os.mkdir(map_folder)
        create_dir(map_folder, subject_list)
        print("Preparing brain representations and inv cov matrix on training data ...")
        main_prepare_mapping(subject_list, betas_folder, explicit_mask_folder, model_name, latent_size, include_bias,
                             map_folder)
    for sub in subject_list:
        ret = False
        if not os.path.exists(opj(map_folder, sub)):
            print(f"Error! Stage 1 does not exist for subject {sub}. Run the script with -m command.")
            ret = True
    if ret:    
        return
    if not os.path.exists(prediction_folder):
        print("Creating output directory for stage 2 ...")
        os.mkdir(prediction_folder)
    print("Prediciting latent vectors for test data ...")
    main_prediction(subject_list, betas_folder_test, nb_test, map_folder, model_name, explicit_mask_folder,
                    include_bias, prediction_folder)


if __name__ == '__main__':
    main()
