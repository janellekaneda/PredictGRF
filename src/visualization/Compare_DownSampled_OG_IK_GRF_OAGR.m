clear all; close all; clc;

% -------------------------------------------------------------------------
% This scripts compares the original and downsampled (and clipped) IK and
% GRF files for the OAGR dataset for a given subject.
% -------------------------------------------------------------------------

%% SETUP

% % PARAMETERS % %
BASE_DIR = 'W:\OA_GaitRetraining\Janelle\CS230\ForUpload';
% % % % % % % % % %

% Import the OpenSim libraries.
import org.opensim.modeling.*;

% Specify base results directory.
resultsdir = fullfile(BASE_DIR, 'data\raw\OAGR_DownSampledFiles\');

% Input subject ID.
subject_id = 'Subject_155';

% Input a trial name for the subject.
trial_name = 'baseline_TM1';

% Input a step number for the trial type.
step_id = '20';

%% PLOT AND COMPARE

subj_dir = fullfile(resultsdir, subject_id);

% Load and compare IK
% [ik_data_og, ik_headers_og] = load_sto(subj_dir, ['Trimmed_' trial_name '_JCs_ik_updated.sto']);
[ik_data_ds, ik_headers_ds] = load_sto(fullfile(subj_dir, trial_name), ['results_ik_step' step_id '_Fs60.sto']);
% 
% figure;
% plot(ik_data_og)
% title('IK original')

figure;
plot(ik_data_ds)
title('IK downsampled')

% Load and compare GRF
% grf_filename_og = ['Trimmed_' trial_name '_grf.mot'];
% grf_osimtable = TimeSeriesTable(fullfile(subj_dir, grf_filename_og));
% grf_data_og = osimTableToStruct(grf_osimtable);

[grf_data_ds, grf_headers_ds] = load_mot(fullfile(subj_dir, trial_name), ['forces_step' step_id '_Fs60.mot']);

% figure;
% grf_fields = fields(grf_data_og);
% for header = 1:length(grf_fields) - 1 % don't plot time
%     plot(grf_data_og.(grf_fields{header}))
%     hold on;
% end
% title('GRF original')

figure;
plot(grf_data_ds)
title('GRF downsampled')
