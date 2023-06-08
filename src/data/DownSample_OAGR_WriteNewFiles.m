close all; clear all; clc;

% -------------------------------------------------------------------------
% This script downsamples the OAGR IK results and ground reaction data from
% 100 Hz to 60 Hz (or whatever new frequency you input), and writes new
% .sto and .mot files, respectively, for each step.
% -------------------------------------------------------------------------

%% SETUP

% % PARAMETERS % %
BASE_DIR = 'W:\OA_GaitRetraining\Janelle\CS230\ForUpload';
% % % % % % % % % %

% Specify subjects to generate for.
%subjs = [102 103 105 linspace(106,137,32)];
%subjs = linspace(138, 169, 32);
%subjs = 101;
subjs = [101 102 103 105 linspace(106,169,64)]; % all 68 subjects

% Specify base data directory.
basedatadir = 'W:\OA_GaitRetraining\OpenSimAllSubjects\';

% Specify base results directory.
resultsdir = fullfile(BASE_DIR, 'data\raw\OAGR_DownSampledFiles\');

Fs_old = 100; % Hz
Fs_new = 60; % Hz; new sampling frequency

%% BATCH DOWNSAMPLE

gait_types = {'baseline_TM1','eval_5deg1','eval_10deg1','eval_neg5deg1','eval_neg10deg1'};
%gait_types = {'baseline_TM1'};

steps_start = 1;
steps_end = 20;
%steps_end = 1;

for i=1:length(subjs)

    % Make subject folder in results directory.
    f = dir(resultsdir);
    if ~isempty(contains({f(:).name}, ['Subject_' num2str(subjs(i))]))
        mkdir(fullfile(resultsdir, ['Subject_' num2str(subjs(i))]));
    end

    for g=1:length(gait_types)

        % Make a gait type folder in results directory.
        f = dir(fullfile(resultsdir, ['Subject_' num2str(subjs(i))]));
        if ~isempty(contains({f(:).name}, gait_types{g}))
            mkdir(fullfile(resultsdir, ['Subject_' num2str(subjs(i))], gait_types{g}));
        end

        % Load original ground reaction .mot file (one file per gait type, covers
        % all steps.
        grf_path = fullfile(basedatadir,['Subject_' num2str(subjs(i))],'Gait','Week1');
        grf_filename = [gait_types{g} '_forces.mot'];
        [grf_data_all, grf_headers] = load_mot(grf_path, grf_filename);

        for s=steps_start:steps_end

            % Load original IK results file (one file per step).
            ik_res_path = fullfile(basedatadir,['Subject_' num2str(subjs(i))],'IK','Results_01_2021_filtered','Week1',gait_types{g});
            ik_res_filename = ['results_ik_step' num2str(s) '.sto'];
            [ik_data, ik_headers] = load_sto(ik_res_path, ik_res_filename);
            
            % Store start and end times based on IK results file for
            % current step.
            start_time = ik_data(1,1);
            end_time = ik_data(end,1);

            % Get GRF data corresponding to IK timestamps.
            grf_time = grf_data_all(:,1);
            start_ix = find(round(grf_time,2)==start_time);
            end_ix = find(round(grf_time,2)==end_time);
            grf_data = grf_data_all(start_ix:end_ix,:);

            % Check that num_rows in original IK data and clipped GRF data
            % match.
            if size(grf_data,1) ~= size(ik_data, 1)
                error('Number of rows in OG IK and clipped GRF data do not match.')
            end

            % Specify old and target time vectors to interpolate the data
            % for.
            time_old = 0:(1/Fs_old):(1/Fs_old)*(size(ik_data, 1)-1);
            time_new = 0:(1/Fs_new):(1/Fs_old)*(size(ik_data, 1)-1);
            
            % Downsample IK data.
            ik_data_new = interp1(time_old, ik_data, time_new);
            % Downsample ground reaction data.
            grf_data_new = interp1(time_old, grf_data, time_new);
  
            % Save new data files: updated IK file, and now a GRF file per
            % step.
            new_data_filepath = fullfile(resultsdir,['Subject_' num2str(subjs(i))],gait_types{g});
            writeSTO_Updated(ik_data_new, ik_headers, new_data_filepath, ['results_ik_step' num2str(s) '_Fs' num2str(Fs_new)],1,1);
            writeMOT_Updated(grf_data_new, grf_headers, new_data_filepath, ['forces_step' num2str(s) '_Fs' num2str(Fs_new)],1,1);

        end
    end
    %disp(['~~~~FINISHED PROCESSING Subject_' num2str(subjs(i)) '~~~~'])
end
