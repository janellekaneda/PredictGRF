clear all; close all; clc;

% This script plots the raw (unfiltered) IK results from the ACL dataset
% and compares to filtered results

%% SETUP

% % PARAMETERS % %
BASE_DIR = 'W:\OA_GaitRetraining\Janelle\CS230\ForUpload';
%BASE_DIR = '/Volumes/HumanPerformanceLab/OA_GaitRetraining/Janelle/CS230/ForUpload';
% % % % % % % % % %

% Import the OpenSim libraries.
import org.opensim.modeling.*;

% Add filtering scripts
addpath("W:\OA_GaitRetraining\Grand Challenge data\Janelle GC Validation Project\Matlab\Julie External Loads Code")

%% PROCESS

% Load data
filedir = "W:\Julie ACL project data\Soccer ACL 10-12 yo\Pre\Control\091014_400\OpenSim output\NEW Not Fixed\IK\IK_w_modelJCs\";
filepath = "Trimmed_RLDJ1_JCs_ik.sto";
[data_og, headers] = load_sto(filedir, filepath);

% Filter data using GRF filter settings
filterProp.Fcut = 30;
filterProp.N = 4;
filterProp.filtType = 'crit';
rate = 200; % Hz; original sampling rate
data_flt = filterDataSet_NEWcorrect(data_og, filterProp, rate);

% Plot key joint angles
joint_angles = {'hip_flexion_r', 'knee_flexion_r', 'ankle_angle_r'};

figure;
for i = 1:length(joint_angles)
    % get ix
    ix = find(strcmp(headers, joint_angles{i}));
    plot(data_og(:,ix));
    hold on;
    plot(data_flt(:,ix));
end

legend('hip og', 'hip filt',...
       'knee og', 'knee filt',...
       'ankle og','ankle filt');
