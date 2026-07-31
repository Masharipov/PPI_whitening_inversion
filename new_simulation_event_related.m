% =========================================================================
% This script extends the original simulation with an event-related design
% and autocorrelated noise.
%
% Event duration, number of events, mean ISI, ISI range, and SNR can be
% specified below. SNR is defined as:
%
%       SNR = SD(noiseless BOLD signal) / SD(noise)
%
% Two conditions are evaluated:
%   1. TR = 0.9 s with AR(4) noise for rapidly sampled fMRI data
%   2. TR = 2.0 s with AR(1) noise
%
% For each condition, whitening inversion is compared with matched
% whitening of both the BOLD signal and the deconvolution matrix.
% =========================================================================

clear;
close all;
clc;

rng(1);

%% ========================================================================
% Simulation parameters
% =========================================================================

event_duration = 1;         % Event duration, s
n_events = 20;              % Number of events

mean_ISI = 5;               % Mean interval between events, s
ISI_range = [2 9];          % Minimum and maximum ISI, s

initial_rest = 5;           % Rest before the first event, s
final_rest = 5;             % Rest after the final event, s

SNR = 3;                    % SD(signal)/SD(noise)
nRep = 100;

lam = 0.005;
burnIn = 500;

% Simulation conditions
TR_values = [0.9 2];
AR_orders = [4 1];

% Coefficients used to generate AR noise
AR_coefficients = { ...
    [0.4 0.2 0.1 0.05], ...    % AR(4), TR = 0.9 s
    0.4};                      % AR(1), TR = 2.0 s


%% ========================================================================
% Generate event-related design
% =========================================================================

if mean_ISI < ISI_range(1) || mean_ISI > ISI_range(2)
    error('mean_ISI must be within ISI_range.');
end

% Generate bounded variable ISIs
ISI = ISI_range(1) + diff(ISI_range)*rand(n_events-1,1);

% Adjust the realized mean while preserving the specified range
for i = 1:100
    ISI = ISI + mean_ISI - mean(ISI);
    ISI = min(max(ISI,ISI_range(1)),ISI_range(2));
end

% ISI is defined from event offset to the next event onset
event_onsets = initial_rest + [0; cumsum(event_duration + ISI)];

duration_sec = event_onsets(end) + event_duration + final_rest;

fprintf('Number of events: %d\n',n_events);
fprintf('Mean ISI: %.3f s\n',mean(ISI));
fprintf('ISI range: %.3f to %.3f s\n',min(ISI),max(ISI));
fprintf('Total duration: %.3f s\n',duration_sec);
fprintf('SNR: %.3f\n\n',SNR);


%% ========================================================================
% Run simulations
% =========================================================================

R = zeros(numel(TR_values),7);

figure('Color','w');

for c = 1:numel(TR_values)

    TR = TR_values(c);
    p = AR_orders(c);
    phi_true = AR_coefficients{c};

    N = ceil(duration_sec/TR);
    t = (0:N-1)'*TR;

    %% Event-related neuronal signal

    timeseries_test = zeros(N,1);

    scan_start = t;
    scan_end = t + TR;

    for e = 1:n_events

        event_start = event_onsets(e);
        event_end = event_start + event_duration;

        overlap = max(0, ...
            min(scan_end,event_end) - ...
            max(scan_start,event_start));

        timeseries_test = timeseries_test + overlap/TR;
    end

    %% Forward model

    hrf = spm_hrf(TR);

    timeseries_conv_tmp = conv(timeseries_test,hrf);
    timeseries_conv = timeseries_conv_tmp(1:N);

    H = convmtx(hrf,N);
    H = H(1:N,:);                         % H: convolution matrix

    truth_z = zscore(timeseries_test);

    %% Repeat simulation with autocorrelated noise

    metrics = zeros(nRep,4);

    curves_inv = zeros(N,nRep);
    curves_matched = zeros(N,nRep);

    for r = 1:nRep

        % Generate autoregressive noise
        innovations = randn(N + burnIn,1);

        noise_all = filter( ...
            1, ...
            [1 -phi_true], ...
            innovations);

        noise = noise_all(burnIn+1:end);

        % Set the selected SNR
        noise = zscore(noise);
        noise = std(timeseries_conv)/SNR*noise;

        noisy_signal = timeseries_conv + noise;

        % Estimate the AR model 
        est_ar = estimate(arima(p,0,0),noisy_signal,'Display','off');

        phi = cell2mat(est_ar.AR);

        % Construct whitening matrix
        A = eye(N);

        for k = 1:p
            A = A - phi(k)*diag(ones(N-k,1),-k);
        end

        V_inv = A*A';
        [eig_vector,eig_value] = eig(V_inv);

        w_mat = real( ...
            eig_vector*sqrtm(eig_value)*eig_vector');

        % Apply whitening to signal and deconvolution matrix
        W_timeseries_conv = w_mat*noisy_signal;
        W_H = w_mat*H;

        % Whitening inversion
        invW_Y = w_mat\W_timeseries_conv;

        ts_decv_invW = deconvolve(invW_Y,N,W_H,lam);

        % Matched whitening
        ts_decv_W = deconvolve(W_timeseries_conv,N,W_H,lam);

        % Standardize
        invW_z = zscore(ts_decv_invW);
        matched_z = zscore(ts_decv_W);

        curves_inv(:,r) = invW_z;
        curves_matched(:,r) = matched_z;

        metrics(r,:) = [ ...
            corr(truth_z,invW_z), ...
            corr(truth_z,matched_z), ...
            sqrt(mean((truth_z-invW_z).^2)), ...
            sqrt(mean((truth_z-matched_z).^2))];
    end

    R(c,:) = [ ...
        TR, ...
        p, ...
        N, ...
        mean(metrics,1)];

    % Select a realization closest to the mean performance
    mean_corr = mean(metrics(:,1:2),1);
    [~,r_plot] = min(sum((metrics(:,1:2)-mean_corr).^2,2));

    %% Plot representative single realization

    subplot(1,2,c);

    plot(t,truth_z,'g','LineWidth',2);
    hold on;

    plot(t,curves_inv(:,r_plot),'b--','LineWidth',1.3);

    plot(t,curves_matched(:,r_plot),'Color',[0.3 0.3 0.3],'LineWidth',1.3);

    hold off;
    axis tight;

    yl = ylim;
    padding = 0.05*diff(yl);
    ylim([yl(1)-padding yl(2)+padding]);

    title(sprintf('TR = %.1f s, AR(%d), SNR = %.1f',TR,p,SNR));

    xlabel('Time, s');
    ylabel('Standardized amplitude');

    if c == 2
        legend( ...
            'Ground truth', ...
            'Whitening inversion', ...
            'Matched whitening', ...
            'Location','best');
    end
end


%% ========================================================================
% Display results
% =========================================================================

results = array2table(R, ...
    'VariableNames',{ ...
        'TR_seconds', ...
        'AR_order', ...
        'N_scans', ...
        'Correlation_inversion', ...
        'Correlation_matched', ...
        'RMSE_inversion', ...
        'RMSE_matched'});

disp(results);


%% ========================================================================
% Original subfunction from He et al. (2025):
% =========================================================================
function ts_decv = deconvolve(timeseries_conv, N, H, lam)    
    % Compute the deconvolved time series
    ts_decv = (H' * H + lam * eye(N)) \ (H' * timeseries_conv);
end