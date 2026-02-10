% =========================================================================
% The first part of this script reproduces the original simulation code
% from He et al. (2025) used to generate Fig. 1b.
%
% The second part of this script extends the original simulation by
% applying whitening to the deconvolution matrix, consistent with the
% spm_PEB formulation.
% =========================================================================

% =========================================================================
% Original code:
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
ts_decv = deconvolve(timeseries_conv,N, H,lam);

% Fit a AR(1) model
ar1 = arima(1,0,0);
est_ar1 = estimate(ar1,timeseries_conv);

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

% =========================================================================
% Update:
% =========================================================================

% Prewhiten signal and deconvolution matrix (as in spm_PEB)
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
% Original subfunction:
% =========================================================================
function ts_decv = deconvolve(timeseries_conv, N, H, lam)    
    % Compute the deconvolved time series
    ts_decv = (H' * H + lam * eye(N)) \ (H' * timeseries_conv);
end


