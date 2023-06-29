close all; clear all; clc;

% -------------------------------------------------------------------------
% This script downsamples the ACL IK results and ground reaction data from
% 200 Hz (IK data) and 2000 Hz (GRF data) to 60 Hz (or whatever new
% frequency you input), and writes new .sto and .mot files, respectively,
% for each trial.
% -------------------------------------------------------------------------

%% SETUP

% % PARAMETERS % %
BASE_DIR = 'W:\OA_GaitRetraining\Janelle\CS230\ForUpload';
% % % % % % % % % %

% Import the OpenSim libraries.
import org.opensim.modeling.*;

% Specify base data directory.
basedatadir = 'W:\Julie ACL project data\';

% Specify base results directory.
resultsdir = fullfile(BASE_DIR, 'data\raw\ACL_DownSampledFiles\');

% Add filtering scripts
addpath("W:\OA_GaitRetraining\Grand Challenge data\Janelle GC Validation Project\Matlab\Julie External Loads Code")

Fs_old_ik = 200; % Hz
Fs_old_grf = 2000; % Hz
Fs_new = 100; % Hz; new sampling frequency

% Run first half (1) or second half (0) of subjs?
first_half_subjs = 0;

%% MORE SETUP

grf_headers = {'time',...
    '1_ground_force_vx','1_ground_force_vy','1_ground_force_vz',...
    '1_ground_force_px','1_ground_force_py','1_ground_force_pz',...
    '2_ground_force_vx','2_ground_force_vy','2_ground_force_vz',...
    '2_ground_force_px','2_ground_force_py','2_ground_force_pz',...
    '3_ground_force_vx','3_ground_force_vy','3_ground_force_vz',...
    '3_ground_force_px','3_ground_force_py','3_ground_force_pz',...
    '1_ground_torque_x','1_ground_torque_y','1_ground_torque_z',...
    '2_ground_torque_x','2_ground_torque_y','2_ground_torque_z',...
    '3_ground_torque_x','3_ground_torque_y','3_ground_torque_z'};

age_group = {'Soccer ACL 10-12 yo', 'Soccer ACL 14-18 yo'};

% Store trial paths that were not updated.
trials_notupdated = {};

% Store trials where original downsampled IK and GRF are diff sizes.
trials_diff_ds_size =  {};

% % Combine IC/FO timing structs. % %

% Load IC/FO times data .mat.
load(fullfile(basedatadir, 'Soccer ACL 10-12 yo', 'MATLAB data structures', 'Events.mat'));
load(fullfile(basedatadir, 'Soccer ACL 14-18 yo', 'MATLAB data structures', 'Events_HS.mat'));

% Initialize TimesStruct, containing all subject info.
TimesStruct = struct();

all_timing_structs = {Events, Events_HS};

for struct = 1:length(all_timing_structs)

    curr_struct = all_timing_structs{struct};
    subj_groupings = fields(curr_struct);

    for subj_grouping = 1:length(subj_groupings)
        subj_names = fields(curr_struct.(subj_groupings{subj_grouping}));
        for subj = 1:length(subj_names)
            subj_struct = curr_struct.(subj_groupings{subj_grouping}).(subj_names{subj});
            TimesStruct.(subj_names{subj}) = subj_struct;

        end
    end
end

%% BATCH PROCESS

% Get all individual subject folder names.
subjs = GetSubDirsFirstLevelOnly(resultsdir);

if first_half_subjs
    subjs = subjs(1:length(subjs)/2);
else
    subjs = subjs((length(subjs)/2 + 1):end);
end

for i = 1:length(subjs)

    subj_dir = fullfile(resultsdir, subjs{i});

    % Get all trial names.
    trial_names = fields(TimesStruct.(['subject_' subjs{i}]));

    for k = 1:length(trial_names)

        % Check if trial name exists in updated IK files.
        ik_res_updated_path = fullfile(subj_dir, [trial_names{k} '_JCs_ik_updated.sto']);

        % Run thru pipeline if updated IK file exists, and have not already
        % generated the clipped & downsampled files (if not want to rerun
        % existing; comment if not).
        %if isfile(ik_res_updated_path) && ~isfile(fullfile(subj_dir,  [trial_names{k} '_JCs_ik_updated' '_Fs' num2str(Fs_new) '.sto'])) && ~isfile(fullfile(subj_dir,  [trial_names{k} '_grf' '_Fs' num2str(Fs_new) '.mot']))
        if isfile(ik_res_updated_path)

            % Load in updated IK results file.
            [ik_data_all, ik_headers] = load_sto(subj_dir, [trial_names{k} '_JCs_ik_updated.sto']);

            % Get IC (initial contact, aka start time) and FO (foot off, aka end time).
            start_time = TimesStruct.(['subject_' subjs{i}]).(trial_names{k}).IC_time;
            end_time = TimesStruct.(['subject_' subjs{i}]).(trial_names{k}).FO_time;

            % Clip IK data.
            ik_time = ik_data_all(:,1);
            start_ix_ik = interp1(ik_time,1:length(ik_time),start_time,'nearest','extrap');
            end_ix_ik = interp1(ik_time,1:length(ik_time),end_time,'nearest','extrap');
            ik_data_raw = ik_data_all(start_ix_ik:end_ix_ik,:);

            % Filter IK data (GRF filter settings)
            filterProp.Fcut = 30;
            filterProp.N = 4;
            filterProp.filtType = 'crit';
            ik_data = filterDataSet_NEWcorrect(ik_data_raw, filterProp, Fs_old_ik);

            % Load in ground reaction data.
            grf_filename_og = [trial_names{k} '_grf.mot'];
            grf_osimtable = TimeSeriesTable(fullfile(subj_dir, grf_filename_og));
            grf_data_all = osimTableToStruct(grf_osimtable);

            % Store ground reaction data in a regular matrix.
            grf_data_mat = zeros(length(grf_data_all.time), length(grf_headers));
            for header = 1:length(grf_headers)
                if header == 1
                    grf_data_mat(:,header) = grf_data_all.time; % don't need to preprend 'unlabeled' for time
                else
                    grf_data_mat(:,header) = grf_data_all.(['unlabeled' grf_headers{header}]);
                end
            end

            % Clip ground reaction data.
            grf_time = grf_data_mat(:,1);
            start_ix_grf = interp1(grf_time,1:length(grf_time),start_time,'nearest','extrap');
            end_ix_grf = interp1(grf_time,1:length(grf_time),end_time,'nearest','extrap');
            grf_data = grf_data_mat(start_ix_grf:end_ix_grf,:);

             % Downsample IK data.
            time_old_ik = 0:(1/Fs_old_ik):(1/Fs_old_ik)*(size(ik_data, 1)-1);
            time_new_ik = 0:(1/Fs_new):(1/Fs_old_ik)*(size(ik_data, 1)-1);
            %ik_data_new = interp1(time_old_ik, ik_data, time_new_ik);

            % Downsample ground reaction data.
            time_old_grf = 0:(1/Fs_old_grf):(1/Fs_old_grf)*(size(grf_data, 1)-1);
            time_new_grf = 0:(1/Fs_new):(1/Fs_old_grf)*(size(grf_data, 1)-1);
            %grf_data_new = interp1(time_old_grf, grf_data, time_new_grf);
            
            % Downsample to smaller new time array.
            if length(time_new_ik) < length(time_new_grf)
                ik_data_new = interp1(time_old_ik, ik_data, time_new_ik);
                grf_data_new = interp1(time_old_grf, grf_data, time_new_ik);
            else % same size or GRF is shorter
                ik_data_new = interp1(time_old_ik, ik_data, time_new_grf);
                grf_data_new = interp1(time_old_grf, grf_data, time_new_grf);
            end
            
            % Check that num_rows match in downsampled IK and GRF data.
            if size(ik_data_new,1) ~= size(grf_data_new,1) % if off, usually off by one
                if size(ik_data_new,1) < size(grf_data_new,1)
                    % Downsample GRF data to match IK data size
                    grf_data_new = interp1(grf_data_new(:,1), grf_data_new, ik_data_new(:,1));
                else % IK dim bigger than GRF dim, so downsample IK data size
                    ik_data_new = interp1(ik_data_new(:,1), ik_data_new, grf_data_new(:,1));
                end
                % Keep track of original data mismatches
                trials_diff_ds_size(end+1) = {[subjs{i} ': ' trial_names{k}]};
                %error('ERROR: Number of rows in downsampled IK and GRF data do not match.')
            end

            % Save new data files.
            writeSTO_Updated(ik_data_new, ik_headers, subj_dir, [trial_names{k} '_JCs_ik_updated' '_Fs' num2str(Fs_new)], 1, 1);
            writeMOT_Updated(grf_data_new, grf_headers, subj_dir, [trial_names{k} '_grf' '_Fs' num2str(Fs_new)], 1, 1);
            %writeSTO_Updated(grf_data_new, grf_headers, subj_dir, [trial_names{k} '_grf' '_Fs' num2str(Fs_new)], 1, 1);

        else
            trials_notupdated(end+1) = {ik_res_updated_path}; % keep track of missing trials
        end

    end

end

%% Output tracking data.
writecell(trials_diff_ds_size, 'DownSample_ACL_trials_diff_ds_size_SecondHalf_FixOutputSize_100Hz.xlsx');
writecell(trials_notupdated, 'DownSample_ACL_trials_notupdated_SecondHalf_FixOutputSize_100Hz.xlsx');
