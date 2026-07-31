% =========================================================================
% Part 1 reproduces the original simulation code
% from He et al. (2025) used to generate Fig. 1b.
%
% Part 2 applies whitening to both the signal and the deconvolution matrix,
% consistent with the spm_peb_ppi implementation.
%
% Part 3 evaluates whether the result depends on temporal sampling.
% The original TR = 0.9 s, AR(1) condition is compared with TR = 2 s, AR(1).
%
% =========================================================================

clear;
close all;
clc;

%% ========================================================================
% Part 1: Original simulation code (He et al., 2025)
% =========================================================================

timeseries_200 = repmat([ones(1,20), zeros(1,20)], 1, 5);
timeseries_test = timeseries_200(1:190);

TR = 0.9; N = 190;
hrf = spm_hrf(TR);

timeseries_conv_tmp = conv(timeseries_test,hrf);
timeseries_conv = timeseries_conv_tmp(1:N)';

H = convmtx(hrf, N);
H = H(1:N, :);                              % H : convolution matrix

lam = 0.002;
ts_decv = deconvolve(timeseries_conv,N,H,lam);

% Fit a AR(1) model
ar1 = arima(1,0,0);
est_ar1 = estimate(ar1,timeseries_conv,'Display','off');

phi = cell2mat(est_ar1.AR);

A = diag(repelem(1,N));
for i = 2:N
    A(i, i-1) = -phi;
end

V_inv = A*A';
[eig_vector, eig_value] = eig(V_inv);

w_mat = eig_vector * sqrtm(eig_value) * eig_vector';

%visualisation
p0 = plot(timeseries_test); axis([0 200 -2.4 3.4]);
set(p0,{'LineWidth','Color'},{2,'g'});
hold on;
p1 = plot(ts_decv,'--');
set(p1,{'LineWidth','Color'},{2,'blue'});
p2 = plot(0.4+10*deconvolve(w_mat*timeseries_conv,N, H,lam),':');
set(p2,{'LineWidth','Color'},{1.3,[0.3 0.3 0.3]});
hold off;
legend('simulated neural signal: ground truth','deconvolved response','prewhiten then deconvolved response');

%% ========================================================================
% Part 2: Apply whitening to both the signal and the deconvolution matrix
% =========================================================================

% Prewhiten signal and deconvolution matrix (as in spm_peb_ppi)
W_timeseries_conv = w_mat*timeseries_conv;
W_H = w_mat*H;

% Whitening inversion prior to deconvolution (as suggested in He et al., 2025)
invW_Y = inv(w_mat)*W_timeseries_conv;

% Deconvolution after whitening inversion
ts_decv_invW = deconvolve(invW_Y,N,W_H,lam);

% Deconvolution without whitening inversion
ts_decv_W = deconvolve(W_timeseries_conv,N,W_H,lam);

% New visualization
figure;
p0 = plot(timeseries_test); axis([0 200 -2.4 3.4]);
set(p0,{'LineWidth','Color'},{2,'g'});
hold on;
p1 = plot(0.2*ts_decv_invW,'--');
set(p1,{'LineWidth','Color'},{2,'blue'});
p2 = plot(0.4+ts_decv_W,':');
set(p2,{'LineWidth','Color'},{1.3,[0.3 0.3 0.3]});
hold off;
legend('simulated neural signal: ground truth','inverse whiten data and prewhiten matrix','prewhiten data and matrix');  

% =========================================================================
% Part 3: Sensitivity to temporal sampling 
% =========================================================================

TR_values = [0.9 2];
R = zeros(numel(TR_values),7);

figure;

for c = 1:numel(TR_values)

    TR = TR_values(c);

    % Preserve the original duration (171 s) and block duration (18 s)
    N = round(171/TR);
    block_N = round(18/TR);

    t = (0:N-1)'*TR;

    timeseries_tmp = repmat( ...
        [ones(1,block_N), zeros(1,block_N)], ...
        1, ceil(N/(2*block_N)) + 1);

    timeseries_test_c = timeseries_tmp(1:N)';

    hrf = spm_hrf(TR);

    timeseries_conv_tmp = conv(timeseries_test_c,hrf);
    timeseries_conv_c = timeseries_conv_tmp(1:N);

    H_c = convmtx(hrf,N);
    H_c = H_c(1:N,:);

    % Fit AR model
    est_ar = estimate(arima(1,0,0),timeseries_conv_c,'Display','off');

    phi = cell2mat(est_ar.AR);

    A = diag(repelem(1,N));
    for i = 2:N
        A(i,i-1) = -phi;
    end

    V_inv = A*A';
    [eig_vector,eig_value] = eig(V_inv);

    w_mat_c = eig_vector*sqrtm(eig_value)*eig_vector';

    % Apply whitening to signal and deconvolution matrix
    W_timeseries_conv = w_mat_c*timeseries_conv_c;
    W_H = w_mat_c*H_c;

    % Whitening inversion
    invW_Y = inv(w_mat_c)*W_timeseries_conv;
    ts_decv_invW = deconvolve(invW_Y,N,W_H,lam);

    % Matched whitening
    ts_decv_W = deconvolve(W_timeseries_conv,N,W_H,lam);

    % Standardize 
    truth_z = zscore(timeseries_test_c);
    invW_z = zscore(ts_decv_invW);
    matched_z = zscore(ts_decv_W);

    R(c,:) = [ ...
        TR, ...
        1, ...
        N, ...
        corr(truth_z,invW_z), ...
        corr(truth_z,matched_z), ...
        sqrt(mean((truth_z-invW_z).^2)), ...
        sqrt(mean((truth_z-matched_z).^2))];

    subplot(1,2,c);

    plot(t,truth_z,'g','LineWidth',2);
    hold on;

    plot(t,invW_z,'b--','LineWidth',1.3);

    plot(t,matched_z,'Color',[0.3 0.3 0.3],'LineWidth',1.3);

    hold off;
    axis tight;

    yl = ylim;
    padding = 0.05*diff(yl);
    ylim([yl(1)-padding, yl(2)+padding]);

    title(sprintf('TR = %.1f s, AR(%d)',TR,1));
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


