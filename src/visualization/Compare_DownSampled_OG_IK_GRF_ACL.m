clear all; close all; clc;

% -------------------------------------------------------------------------
% This scripts compares the original and downsampled (and clipped) IK and
% GRF files for the ACL dataset for a given subject.
% -------------------------------------------------------------------------

%% SETUP

% % PARAMETERS % %
BASE_DIR = 'W:\OA_GaitRetraining\Janelle\CS230\ForUpload';
%BASE_DIR = '/Volumes/HumanPerformanceLab/OA_GaitRetraining/Janelle/CS230/ForUpload';
% % % % % % % % % %

% Import the OpenSim libraries.
import org.opensim.modeling.*;

% Specify base results directory.
resultsdir = fullfile(BASE_DIR, 'data', 'raw', 'ACL_DownSampledFiles');

% Input subject ID.
subject_id = '091614_330';

% Input a trial name for the subject.
trial_name = 'LLDJ3';

%% PLOT AND COMPARE

subj_dir = fullfile(resultsdir, subject_id);

%%
% % Load and compare IK
% [ik_data_og, ik_headers_og] = load_sto(subj_dir, ['Trimmed_' trial_name '_JCs_ik_updated.sto']);
[ik_data_ds, ik_headers_ds] = load_sto(subj_dir, ['Trimmed_' trial_name '_JCs_ik_updated_Fs100.sto']);
% 
% figure;
% plot(ik_data_og)
% %legend(ik_headers_og)
% title('IK original')
% 
figure;
plot(ik_data_ds)
%legend(ik_headers_ds)
title('IK downsampled')
%%
% Load and compare GRF
grf_filename_og = ['Trimmed_' trial_name '_grf.mot'];
grf_osimtable = TimeSeriesTable(fullfile(subj_dir, grf_filename_og));
grf_data_og = osimTableToStruct(grf_osimtable);

[grf_data_ds, grf_headers_ds] = load_mot(subj_dir, ['Trimmed_' trial_name '_grf_Fs100.mot']);

% % Compare Fy waveforms from each plate % %
% Original
grf_og_fy_names = {'unlabeled1_ground_force_vy';'unlabeled2_ground_force_vy';'unlabeled3_ground_force_vy'};
figure; plot(grf_data_og.(grf_og_fy_names{1})); hold on; plot(grf_data_og.(grf_og_fy_names{2})); hold on; plot(grf_data_og.(grf_og_fy_names{3})); legend('plate1','plate2','plate3');
% Downsampled
figure; plot(grf_data_ds(:,3)); hold on; plot(grf_data_ds(:,9)); hold on; plot(grf_data_ds(:,15)); legend('plate1','plate2','plate3');

% figure;
% % grf_fields = fields(grf_data_og);
% % for header = 1:length(grf_fields) - 1 % don't plot time
% %     plot(grf_data_og.(grf_fields{header}))
% %     hold on;
% % end
% grf_fields = fields(grf_data_og);
% for header = 1:length(grf_fields)
%     plot(grf_data_og.(grf_fields{header}))
%     hold on;
% end
% title('GRF original')
% 
% figure;
% plot(grf_data_ds)
% legend(grf_headers_ds)
% title('GRF downsampled')
