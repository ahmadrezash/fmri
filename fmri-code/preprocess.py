"""
# Preprocessing Workflow

In this workflow we will conduct the following steps:

It's always good to know which interfaces you want to use in your workflow and in which order you want to execute them.
 For the preprocessing workflow, we use the following nodes:

0. Gunzip (nipype)
1. Slice Timing Correction (SPM) `from nipype.interfaces.spm import SliceTiming`
2. Realign (SPM) `from nipype.interfaces.spm import Realign`
3. Coregistration (SPM) `from nipype.interfaces.spm import Coregister`
"""
import json
import os
import warnings
from os.path import join as opj

import nipype.interfaces.matlab as mlab
import numpy as np
from nipype import MapNode
from nipype.algorithms.misc import Gunzip
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.spm import SliceTiming, Realign, Coregister
from nipype.interfaces.utility import IdentityInterface
from nipype.pipeline.engine import Workflow, Node

from nipype.interfaces import spm
matlab_cmd = '/home/benjamin/spm12/standalone/run_spm12.sh /usr/local/MATLAB/MATLAB_Runtime/v92 script'
spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)

warnings.filterwarnings('ignore')

##############
# Set SPM Path
##############
# mlab.MatlabCommand.set_default_paths('/home/benjamin/spm12/')

if __name__ == '__main__':
    # List of subjects
    subject_list = ['sub-0' + str(n) for n in range(1, 6)]
    print(subject_list)

    # List of sessions for each subject

    ##############
    # Dataset Root
    ##############
    data_root = '/mnt/HD2/benjamin/fmri_recon/dataset/ds001246-download/'

    session_list = dict()
    for subj in subject_list:
        sess = []
        folders = os.listdir(opj(data_root, subj))
        for folder in folders:
            if folder != 'ses-anatomy':
                sess.append(folder)
        session_list[subj] = sess
    print(session_list)

    # List of run files

    run_list = dict()
    for subj in subject_list:
        for ses in session_list[subj]:
            runs = []
            files = os.listdir(opj(data_root, subj, ses, 'func'))
            for file in files:
                if file.endswith("bold.nii.gz"):
                    runs.append(opj(data_root, subj, ses, 'func', file[:-12]))
            run_list[(subj, ses)] = runs
    print(run_list)

    # fMRI parameters
    # TR of functional images

    TR = 3.0
    voxel_size = (3, 3, 3)
    number_of_slices = 50

    ##############
    # JSON Files
    ##############

    json_file1 = "/mnt/HD2/benjamin/fmri_recon/dataset/ds001246-download/task-imagery_bold.json"
    json_file2 = "/mnt/HD2/benjamin/fmri_recon/dataset/ds001246-download/task-perception_bold.json"

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

    slice_order = list(sorted1 + 1)
    print(slice_order)

    ## Specify `Node`s for the main workflow
    gunzip_func_node = MapNode(Gunzip(), iterfield=['in_file'],
                               name='Gunzip_func')
    gunzip_anat_node = Node(Gunzip(),
                            name='Gunzip_anat')

    # Slice Timing Correction

    stc_node = Node(SliceTiming(num_slices=number_of_slices,
                                time_repetition=TR,
                                time_acquisition=TR - (TR / number_of_slices),
                                slice_order=slice_order,
                                ref_slice=50,
                                out_prefix='a'),
                    name='STC')

    # Realig

    realign_node = Node(Realign(register_to_mean=True,
                                out_prefix='r'),
                        name='Realign')

    # Coregister

    coregiter_node = Node(Coregister(out_prefix='ra'),
                          name='Coregister')
    # target (ref in spm)
    # source (source in spm)
    # apply_to_files (other in spm)

    # Specify input & output stream
    # Iterate over subjects
    # Infosource - a function free node to iterate over the list of subject names
    infosrc_subjects = Node(IdentityInterface(fields=['subject_id']),
                            name="infosrc_subjects")
    infosrc_subjects.iterables = [('subject_id', subject_list)]

    infosrc_sessions = Node(IdentityInterface(fields=['session_id']),
                            name="infosrc_sessions")
    infosrc_sessions.itersource = ("infosrc_subjects", 'subject_id')
    infosrc_sessions.iterables = [('session_id', session_list)]

    # SelectFiles
    anat_file = opj('{subject_id}', 'ses-anatomy', 'anat', '{subject_id}_ses-anatomy_T1w.nii.gz')
    # func_file = opj('{subject_id}','ses-[ip]*','func','*bold.nii.gz')
    func_file = opj('{subject_id}', '{session_id}', 'func', '*bold.nii.gz')
    templates = {'anat': anat_file,
                 'func': func_file}

    selectfiles_anat = Node(SelectFiles({'anat': anat_file},
                                        base_directory='/mnt/HD2/benjamin/fmri_recon/dataset/ds001246-download/'),
                            name="selectfiles_anat")
    selectfiles_sessions = Node(SelectFiles({'func': func_file},
                                            base_directory='/mnt/HD2/benjamin/fmri_recon/dataset/ds001246-download/'),
                                name="selectfiles_sessions")

    # Datasink - creates output folder for important outputs
    datasink_node = Node(DataSink(base_directory='/mnt/HD2/benjamin/fmri_recon/',
                                  container='datasink'),
                         name="datasink")
    substitutions = [('_subject_id_', '')]
    datasink_node.inputs.substitutions = substitutions

    # Specify the workflow
    # create the workflow
    preproc = Workflow(name='preprocess')
    preproc.base_dir = '/mnt/HD2/benjamin/fmri_recon/'

    # connect infosource to selectfile
    preproc.connect([(infosrc_subjects, selectfiles_anat, [('subject_id', 'subject_id')])])
    preproc.connect([(infosrc_subjects, infosrc_sessions, [('subject_id', 'subject_id')])])
    preproc.connect([(infosrc_sessions, selectfiles_sessions, [('session_id', 'session_id')])])
    preproc.connect([(infosrc_subjects, selectfiles_sessions, [('subject_id', 'subject_id')])])

    # connect selectfile to gunzip
    preproc.connect([(selectfiles_anat, gunzip_anat_node, [('anat', 'in_file')])])
    preproc.connect([(selectfiles_sessions, gunzip_func_node, [('func', 'in_file')])])

    # connect gunzip_func to slicetiming
    preproc.connect([(gunzip_func_node, stc_node, [('out_file', 'in_files')])])

    # connect to realign
    preproc.connect([(stc_node, realign_node, [('timecorrected_files', 'in_files')])])

    # connect to coregister
    preproc.connect([(realign_node, coregiter_node, [('mean_image', 'source')])])
    preproc.connect([(realign_node, coregiter_node, [('realigned_files', 'apply_to_files')])])
    preproc.connect([(gunzip_anat_node, coregiter_node, [('out_file', 'target')])])

    # keeping realignment params
    preproc.connect([(realign_node, datasink_node, [('realignment_parameters', 'preprocess.@par')])])
    
    preproc.write_graph(graph2use='flat', format='png', simple_form=True)
    preproc.write_graph(graph2use='colored', format='png', simple_form=True)

    ################################
    # Use more processors if you can
    ################################
    preproc.run('MultiProc', plugin_args={'n_procs': 32})
