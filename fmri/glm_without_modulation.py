import sys
import os
from os.path import join as opj

import json

import nipype.algorithms.modelgen as modelgen
from nipype.algorithms.modelgen import SpecifySPMModel
from nipype.algorithms.misc import Gunzip

import nipype.interfaces.spm as spm
from nipype.interfaces.spm import Level1Design, EstimateModel, EstimateContrast
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink

from nipype.pipeline.engine import Workflow, Node, MapNode

from nipype import MapNode

import nipype.interfaces.matlab as mlab
matlab_cmd = '/home/benjamin/spm12/standalone/run_spm12.sh /usr/local/MATLAB/MATLAB_Runtime/v92 script'
spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)

import numpy as np

#####################################################
# Helper function to read tsv files
# This function cannot use anything from global scope
#####################################################
def read_tsv_train(tsv_files):
    import csv
    import nipype
    import numpy as np
    from nipype.interfaces.base import Bunch
    condition_names = ['rest', 'oneback', 'stimulus']
    
    subject_info = []
    for i in range(len(tsv_files)):
        onsets = {'rest':[], 'stimulus':[], 'oneback':[]}
        durs   = {'rest':[], 'stimulus':[], 'oneback':[]}
        amps   = {'rest':[], 'stimulus':[], 'oneback':[]}
        with open(tsv_files[i]) as tsv:
            reader = list(csv.DictReader(tsv, skipinitialspace=True, delimiter=str('\t')))
            for idx in range(len(reader)):
                cond = reader[idx]['event_type']
                if cond == 'stimulus' and idx > 0:
                    if reader[idx]['stim_id'] == reader[idx - 1]['stim_id']:     #event idx is a oneback
                        cond = 'oneback'
                        
                onsets[cond].append(float(reader[idx]['onset']))
                durs[cond].append(float(reader[idx]['duration']))
                amps[cond].append(1.0)
        subject_info.append(Bunch(conditions=condition_names,
                         onsets=[onsets[k] for k in condition_names],
                         durations=[durs[k] for k in condition_names],
                         amplitudes=[amps[k] for k in condition_names],
                         tmod=None,
                         pmod=None,
                         regressor_names=None,
                         regressors=None))
    return subject_info

def read_tsv_test(tsv_files):
    import csv
    import nipype
    import numpy as np
    from nipype.interfaces.base import Bunch
    
    condition_names = ['rest', 'oneback'] + ['1443537.022563', '1621127.019020', '1677366.018182', '1846331.017038', '1858441.011077', '1943899.024131', '1976957.013223', '2071294.046212', '2128385.020264', '2139199.010398', '2190790.015121', '2274259.024319', '2416519.012793', '2437136.012836', '2437971.005013', '2690373.007713', '2797295.015411', '2824058.018729', '2882301.014188', '2916179.024850', '2950256.022949', '2951358.023759', '3064758.038750', '3122295.031279', '3124170.013920', '3237416.058334', '3272010.011001', '3345837.012501', '3379051.008496', '3452741.024622', '3455488.028622', '3482252.022530', '3495258.009895', '3584254.005040', '3626115.019498', '3710193.022225', '3716966.028524', '3761084.043533', '3767745.000109', '3941684.021672', '3954393.010038', '4210120.009062', '4252077.010859', '4254777.016338', '4297750.025624', '4387400.016693', '4507155.021299', '4533802.019479', '4554684.053399', '4572121.003262']
    
    subject_info = []
    for i in range(len(tsv_files)):
        onsets = {k:[] for k in condition_names}
        durs   = {k:[] for k in condition_names}
        amps   = {k:[] for k in condition_names}
        with open(tsv_files[i]) as tsv:
            reader = list(csv.DictReader(tsv, skipinitialspace=True, delimiter=str('\t')))
            for idx in range(len(reader)):
                cond = reader[idx]['event_type']
                if cond == 'stimulus' and idx > 0:
                    if reader[idx]['stim_id'] == reader[idx - 1]['stim_id']:     #event idx is a oneback
                        cond = 'oneback'
                    else:
                        cond = reader[idx]['stim_id']
                        
                onsets[cond].append(float(reader[idx]['onset']))
                durs[cond].append(float(reader[idx]['duration']))
                amps[cond].append(1.0)
        subject_info.append(Bunch(conditions=condition_names,
                         onsets=[onsets[k] for k in condition_names],
                         durations=[durs[k] for k in condition_names],
                         amplitudes=[amps[k] for k in condition_names],
                         tmod=None,
                         pmod=None,
                         regressor_names=None,
                         regressors=None))
    return subject_info

#################
# Input Arguments
#################
import argparse

parser = argparse.ArgumentParser(description="Performs GLM without parametric modulation. It works on top of Kamitani's preprocessed data.")
parser.add_argument('project_folder', type=str, help='address of the project folder containing these directories: dataset, preprocess, datasink')
parser.add_argument('-s', '--subjects', metavar='N', type=str, nargs='+',
                   help='sequence of subjects in the form of sub-01 sub-02 ...', default=['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05'])
parser.add_argument('-t', '--test', dest='training', action='store_const',
                   const=True, default=False,
                   help='performs GLM on test sessions')
parser.add_argument('-p', '--nb_processor', dest='nb_prc', type=int, default=32,
                   help='Number of processors to use')


args = parser.parse_args()

project_folder = args.project_folder
subject_list = args.subjects
training = args.training
nb_prc = args.nb_prc

def main():
    
    #######################
    # Commandline Arguments
    #######################
    # list of subject identifiers
    task_name = "Training" if training else "Test"
    print(project_folder, subject_list, task_name, nb_prc)
    
    #############################################################
    # Extracting fMRI Params (Only works with Kamitani's Dataset)
    #############################################################
    TR = 3.0
    voxel_size = (3,3,3)
    number_of_slices = 50
    json_file1 = opj(project_folder, "dataset/ds001246-download/task-imagery_bold.json")
    json_file2 = opj(project_folder, "dataset/ds001246-download/task-perception_bold.json")

    file = open(json_file1)
    data = json.load(file)
    slice_timing1 = data['SliceTiming']
    file.close()

    file = open(json_file2)
    data = json.load(file)
    slice_timing2 = data['SliceTiming']
    file.close()

    sorted1 = np.argsort(slice_timing1)
    sorted2 = np.argsort(slice_timing2)
    print(np.all(sorted1 == sorted2))

    slice_order = list(sorted1+1)
    print("Slice order:", slice_order)
    
    ##########################
    # Creating essential nodes
    ##########################
    # Model Spec
    modelspec_node = Node(SpecifySPMModel(concatenate_runs=True,
                                     input_units='secs',
                                     output_units='secs',
                                     time_repetition=TR,
                                     high_pass_filter_cutoff=128),
                          name='modelspec')
    
    # Level1Design - Generates a SPM design matrix
    level1design_node = Node(Level1Design(bases={'hrf': {'derivs': [0, 0]}},
                                     timing_units='secs',
                                     interscan_interval=TR,
                                     model_serial_correlations='AR(1)',
                                     mask_threshold='-Inf'),
                        name="level1design")
    
    # EstimateModel - estimate the parameters of the model (GLM)
    level1estimate_node = Node(EstimateModel(estimation_method={'Classical': 1}),
                          name="level1estimate")
    
    # Infosource - a function free node to iterate over the list of subject names
    infosrc_subjects = Node(IdentityInterface(fields=['subject_id']),
                      name="infosrc_subjects")
    infosrc_subjects.iterables = [('subject_id', subject_list)]
    
    # SelectFiles - it select files based on template matching
    tsv_file = opj('dataset', 'ds001246-download', '{subject_id}', 'ses-p*' + task_name + '*', 'func', '{subject_id}_ses-p*' + task_name + '*_task-*_events.tsv')
    reg_file = opj('preprocess', '_subject_id_{subject_id}', '_session_id_ses-p*' + task_name + '*', 'Realign', 'rp_a{subject_id}_ses-p*' + task_name + '*_task-*_bold.txt')
    func_file = opj('preprocess', '_subject_id_{subject_id}', '_session_id_ses-p*' + task_name + '*', 'Coregister', 'rara{subject_id}_ses-p*' + task_name + '*_task-*_bold.nii')
    mask_file = opj('datasink', 'preprocessed_masks', '{subject_id}', '{subject_id}_full_mask.nii')

    templates = {    'tsv' : tsv_file,
                     'reg' : reg_file,
                    'func' : func_file,
                     'mask': mask_file}

    selectfiles = Node(SelectFiles(templates, base_directory=project_folder), name="selectfiles")

    # Subject Info
    subject_info_node = Node(Function(input_names=['tsv_files'], output_names=['subject_info'],
                                      function=read_tsv_train if training else read_tsv_test),
                             name='subject_info')

    # Datasink - creates output folder for important outputs
    datasink_node = Node(DataSink(base_directory=project_folder, container='datasink'), name="datasink")

    substitutions = [('_subject_id_', '')]
    datasink_node.inputs.substitutions = substitutions
                       
    
    #####################
    # Create the workflow
    #####################
    wf_name = 'glm_train_nomod' if training else 'glm_test'
    glm = Workflow(name=wf_name)
    glm.base_dir = project_folder

    # connect infosource to selectfile
    glm.connect([(infosrc_subjects, selectfiles,  [('subject_id', 'subject_id')])])
    glm.connect([(selectfiles, subject_info_node, [('tsv', 'tsv_files')])])

    # connect infos to modelspec
    glm.connect([(subject_info_node, modelspec_node, [('subject_info', 'subject_info')])])
    glm.connect([(selectfiles, modelspec_node, [('reg', 'realignment_parameters')])])
    glm.connect([(selectfiles, modelspec_node, [('func', 'functional_runs')])])

    # connect modelspec to level1design
    glm.connect([(modelspec_node, level1design_node, [('session_info', 'session_info')])])
    glm.connect([(selectfiles, level1design_node, [('mask', 'mask_image')])])

    # connect design to estimate
    glm.connect([(level1design_node, level1estimate_node, [('spm_mat_file', 'spm_mat_file')])])

    # keeping estimate files params
    glm.connect([(level1estimate_node, datasink_node, [('mask_image',     f'{wf_name}.@mask_img')])])
    glm.connect([(level1estimate_node, datasink_node, [('beta_images',    f'{wf_name}.@beta_imgs')])])
    glm.connect([(level1estimate_node, datasink_node, [('residual_image', f'{wf_name}.@res_img')])])
    glm.connect([(level1estimate_node, datasink_node, [('RPVimage',       f'{wf_name}.@rpv_img')])])
    glm.connect([(level1estimate_node, datasink_node, [('spm_mat_file',   f'{wf_name}.@spm_mat_file')])])

    glm.write_graph(graph2use='flat', format='png', simple_form=True)
#     from IPython.display import Image
#     Image(filename=opj(glm.base_dir, {wf_name}, 'graph_detailed.png'))
    
    ##################
    # Run the workflow
    ##################
    glm.run('MultiProc', plugin_args={'n_procs': nb_prc})

if __name__ == "__main__":
    main()