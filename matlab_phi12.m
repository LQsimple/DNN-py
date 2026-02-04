function phi12()
% MATLAB版本的phi代码

% 参数设置
M = 64; N = 64;
Delta_f = 156250;
T = 1 / Delta_f;
MN = M * N;
K = 3; % 目标数

% 模拟目标参数
h_true = [1; 1; 1];
delay_bins = [3; 2; 7]; % 延迟bin (注意：这里保持原值，但在计算时会调整索引)
doppler_bins = [10; 5; 2]; % 多普勒bin
delays = delay_bins / (M * Delta_f); % 实际延迟（秒）
dopplers = doppler_bins / (N * T); % 实际多普勒（Hz）

% 构造发射信号 x = (F_N^H ⊗ G_t) d_dd
rng(42); % 设置随机种子
G_t = eye(M); % 假设为单位矩阵
modSize = 4;
% d_dd = randn(MN, 1) + 1i * randn(MN, 1); % 随机导频
dataBits = randi([0 1], M * N, log2(modSize));
dataDe = bi2de(dataBits);
dataDe = reshape(dataDe, M, N);
data = qammod(dataDe, modSize, 'UnitAveragePower',true);
Xdd = data;
% x = kron(F_N_H, G_t) * d_dd; % 发射信号 x
y_chan = zeros(M * N, 1);
% 构造信道矩阵 H_ni 和字典 Psi

% 检查文件是否存在
if ~exist('12phi.mat', 'file')
    % 如果文件不存在，生成数据
    Psi = build_dictionary(M, N, Delta_f, T, G_t, Xdd);
else
    load('12phi.mat', 'Psi'); % 从文件中加载字典矩阵
    fprintf('12phi.mat文件已存在\n');
end

% 构造稀疏向量 h：非零位置对应目标延迟-多普勒单元
h_sparse = zeros(25 * 25, 1);
% 注意：Matlab使用1-based索引，但这里的计算逻辑保持原样
linear_indices = delay_bins * 25 + doppler_bins + 1; % 调整为1-based索引
h_sparse(linear_indices) = h_true;
Psi = build_dictionary(M, N, Delta_f, T, G_t, Xdd);
% 计算两种 y
y_dict = Psi * h_sparse; % 字典法
for k = 1:K
y_chan = y_chan + OTFS_output(Xdd, T, delays(k), dopplers(k));
end

% 验证是否相等
error_norm = norm(y_dict - y_chan);
fprintf('两种方法构造的 y 的差异范数: %.2e\n', error_norm);
is_equal = norm(y_dict - y_chan) < 1e-10;
fprintf('是否相等: %s\n', mat2str(is_equal));

% 计算Psi每一列的L2范数
% 计算每一列的 L2 范数
col_norms = sqrt(sum(abs(Psi).^2, 1)); % 对复数也适用

% 计算均值和方差
mean_norm = mean(col_norms);
variance_norm = var(col_norms);

% 显示结果
fprintf('L2范数均值: %.4f\n', mean_norm);
fprintf('L2范数方差: %.4f\n', variance_norm);







% 随机选择保留的行索引 (不重复)
n_keep = 25 * 25; % 保留的行数
% kept_indices = randsample(size(Psi, 1), n_keep, false);
kept_indices = datasample(1:size(Psi, 1), n_keep, 'Replace', false);
kept_indices = sort(kept_indices); % 按原顺序排列

% 提取保留的行
reduced_Psi = Psi(kept_indices, :);
reduced_y = y_dict(kept_indices);
reduced_y2 = y_chan(kept_indices);

% 计算每一列的 L2 范数
col_norms = sqrt(sum(abs(reduced_Psi).^2, 1)); % 对复数也适用
% 计算均值和方差
mean_norm = mean(col_norms);
variance_norm = var(col_norms);
% 显示结果
fprintf('提取后L2范数均值: %.4f\n', mean_norm);
fprintf('提取后L2范数方差: %.4f\n', variance_norm);


% 提取后的验证是否相等
error_norm = norm(reduced_y - reduced_y2);
fprintf('提取后的两种方法构造的 y 的差异范数: %.2e\n', error_norm);
is_equal = norm(reduced_y - reduced_y2) < 1e-10;
fprintf('是否相等: %s\n', mat2str(is_equal));

% 将复数矩阵转换为实数系统
[A_real, y_real] = complex_to_real_system(reduced_Psi, reduced_y);

% 保存字典phi
    save('12phi.mat', 'A_real', 'reduced_Psi','kept_indices');
    fprintf('已生成新的12phi.mat文件\n');


% 显示结果信息
fprintf('数据处理完成\n');

fprintf('字典矩阵 Psi 大小: %d x %d\n', size(Psi, 1), size(Psi, 2));
fprintf('字典矩阵 reduced_Psi 大小: %d x %d\n', size(reduced_Psi, 1), size(reduced_Psi, 2));
fprintf('实数矩阵 A_real 大小: %d x %d\n', size(A_real, 1), size(A_real, 2));

end

function ydd = OTFS_output(Xdd, T, delay, Doppler)
[M, N] = size(Xdd);
lt = ceil(delay / (T / M));
deltaf = 1 / T;
Xtf = ISFFT(Xdd, M, N);
rt = exp(1j * 2 * pi * Doppler * (0:1:(M*N - 1))' * T / M) .* circshift(reshape(circshift(ifft(diag(exp(-1j * 2 * pi * (0:1:(M-1)) *  deltaf * delay)) * Xtf ) * sqrt(M), - lt ), [], 1), lt);
Rt = reshape(rt, M, N);
Ydd = fft(Rt.').' / sqrt(N);
ydd = Ydd(:);
end

function tfSignal = ISFFT(ddSignal, M, N)
tfSignal = fft(ifft(ddSignal.').') * sqrt(N) / sqrt(M);
end

function Psi = build_dictionary(M, N, Delta_f, T, G_t, Xdd)
% 构造稀疏字典矩阵 Psi，8x8的情况
MN = M * N;
Psi = zeros(MN, 25 * 25);

for m = 1:25
    for n = 1:25
        col_idx = (m-1) * 25 + n;
        delays = (m-1) / (M * Delta_f); % 实际延迟（秒）
        dopplers = (n-1) / (N * T); % 实际多普勒（Hz）
        temp = OTFS_output(Xdd, T, delays, dopplers);
        Psi(:, col_idx) = temp;
    end
end
end

function varargout = complex_to_real_system(A_complex, y_complex, varargin)
% 将复数线性系统转换为实数系统（按式4.37-4.39）
%
% 参数:
%     A_complex: 复数矩阵 [M, N]
%     y_complex: 复数观测向量 [M,]
%     varargin: 可选的复数解向量 h_complex [N,] 和复数噪声向量 w_complex [M,]
%
% 返回:
%     A_real: 实数矩阵 [2M, 2N]
%     y_real: 实数观测向量 [2M,]
%     h_real: 实数解向量 [2N,] (若h_complex提供)
%     w_real: 实数噪声向量 [2M,] (若w_complex提供)

% 拆分实部和虚部
Re_A = real(A_complex);
Im_A = imag(A_complex);

% 构建实数矩阵 A_real (式4.39)
A_real = [Re_A, -Im_A; Im_A, Re_A];

% 构建实数向量 y_real (式4.38)
y_real = [real(y_complex(:)); imag(y_complex(:))];

% 输出参数
varargout{1} = A_real;
varargout{2} = y_real;

% 处理可选参数
if nargin >= 3 && ~isempty(varargin{1})
    h_complex = varargin{1};
    h_real = [real(h_complex(:)); imag(h_complex(:))];
    varargout{3} = h_real;
end

if nargin >= 4 && ~isempty(varargin{2})
    w_complex = varargin{2};
    w_real = [real(w_complex(:)); imag(w_complex(:))];
    varargout{4} = w_real;
end
end

function F = dft_matrix(N)
% 构造DFT矩阵
% F(k,n) = exp(-2i*pi*(k-1)*(n-1)/N), k,n = 1,2,...,N

k = (0:N-1)';
n = 0:N-1;
F = exp(-2i * pi * k * n / N);
end