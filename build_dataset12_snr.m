function build_dataset12(SNR_values, folder_name)
% MATLAB版本的数据集构建代码 - 支持多SNR值生成
% 
% 输入参数:
%   SNR_values - 信噪比值数组，例如 [-15, -10, -5, 0, 5, 10, 15]
%   folder_name - 保存文件夹名称，默认为'SNR_datasets'
% 
% 默认行为: 如果不输入参数，则生成默认SNR值的数据集

% 参数设置
if nargin < 2
    folder_name = 'SNR_datasets'; % 默认文件夹名称
end

if nargin < 1
    SNR_values = [ -5, 0, 5, 10, 15]; % 默认SNR值
end

M = 64; N = 64;
Delta_f = 156250;
T = 1 / Delta_f;
MN = M * N;
K = 5; % 目标数
num = 1000;
fc = 77e9; % carrier frequency
c0 = physconst('LightSpeed');

delta_R = c0 / (2 * M * Delta_f);  %距离分辨率
delta_tau = 1 / (M * Delta_f);      %延迟分辨率
R_max = M * delta_R;                                                                                                                                                                                                                                                
delta_V = c0 / (2 * fc * N * T); %速度分辨率
delta_fd = 1 / (N * T);            %多普勒分辨率
V_max = delta_V * N/2;
B = M * Delta_f;

% 构造发射信号 x = (F_N^H ⊗ G_t) d_dd
G_t = eye(M); % 假设为单位矩阵
modSize = 4;
dataBits = randi([0 1], M * N, log2(modSize));
dataDe = bi2de(dataBits);
dataDe = reshape(dataDe, M, N);
data = qammod(dataDe, modSize, 'UnitAveragePower', true);
Xdd = data;

% 检查文件是否存在
if ~exist('12phi.mat', 'file')
    error('文件 %s 不存在，程序终止。', '12phi.mat');
else
    load('12phi.mat', 'reduced_Psi'); % 从文件中加载字典矩阵
    fprintf('12phi.mat文件已存在\n');
end

% 随机选择保留的行索引 (不重复)
load('12phi.mat', 'kept_indices');

% 确保SNR_values是行向量
SNR_values = reshape(SNR_values, 1, []);

% 创建文件夹（如果不存在）
if ~exist(folder_name, 'dir')
    mkdir(folder_name);
    fprintf('创建文件夹: %s\n', folder_name);
else
    fprintf('文件夹已存在: %s\n', folder_name);
end

% 显示将要生成的SNR值
fprintf('\n将要为以下SNR值生成数据集: %s\n', mat2str(SNR_values));
fprintf('所有数据集将保存在文件夹: %s\n\n', folder_name);

% 为每个SNR值生成数据集
for idx = 1:length(SNR_values)
    SNRdB = SNR_values(idx);
    fprintf('=== 正在为 SNR = %d dB 生成数据集 (%d/%d) ===\n', SNRdB, idx, length(SNR_values));
    
    % 生成样本
    samples = generate_samples(num, K, Xdd, kept_indices, M, N, Delta_f, T, reduced_Psi, SNRdB);
    
    % 保存数据集到同一文件夹，文件名包含SNR值
    save_filename = fullfile(folder_name, sprintf('dataset_SNR_%d.mat', SNRdB));
    save(save_filename, 'samples', 'kept_indices', 'reduced_Psi');
    fprintf('数据集已保存到: %s\n', save_filename);
end

fprintf('\n========================================\n');
fprintf('所有数据集生成完成！\n');
fprintf('========================================\n');
fprintf('数据集保存位置: %s\n', folder_name);
fprintf('生成的数据集文件:\n');
for idx = 1:length(SNR_values)
    fprintf('  - dataset_SNR_%d.mat (SNR = %d dB)\n', SNR_values(idx), SNR_values(idx));
end

end

function ydd = OTFS_output(Xdd, T, delay, Doppler, SNRdB)
[M, N] = size(Xdd);
lt = ceil(delay / (T / M));
deltaf = 1 / T;
Xtf = ISFFT(Xdd, M, N);
rt = exp(1j * 2 * pi * Doppler * (0:1:(M*N - 1))' * T / M) .* ...
     circshift(reshape(circshift(ifft(diag(exp(-1j * 2 * pi * (0:1:(M-1)) * deltaf * delay)) * Xtf ) * sqrt(M), -lt), [], 1), lt);
rt = awgn(rt, SNRdB, 'measured');
Rt = reshape(rt, M, N);
Ydd = fft(Rt.').' / sqrt(N);
ydd = Ydd(:);
end

function tfSignal = ISFFT(ddSignal, M, N)
tfSignal = fft(ifft(ddSignal.').') * sqrt(N) / sqrt(M);
end

function varargout = complex_to_real_system(y_complex, varargin)
% 将复数线性系统转换为实数系统（按式4.37-4.39）
%
% 参数:
%     y_complex: 复数观测向量 [M,]
%     varargin: 可选的复数解向量 h_complex [N,] 和复数噪声向量 w_complex [M,]

% 构建实数向量 y_real (式4.38)
y_real = [real(y_complex(:)); imag(y_complex(:))];

% 输出参数
varargout{1} = y_real;

% 处理可选参数
if nargin >= 2 && ~isempty(varargin{1})
    h_complex = varargin{1};
    h_real = [real(h_complex(:)); imag(h_complex(:))];
    varargout{3} = h_real;
end

if nargin >= 3 && ~isempty(varargin{2})
    w_complex = varargin{2};
    w_real = [real(w_complex(:)); imag(w_complex(:))];
    varargout{4} = w_real;
end
end

function samples = generate_samples(num_samples, K, Xdd, kept_indices, M, N, Delta_f, T, Psi, SNRdB)
% 生成样本数据
samples = cell(num_samples, 4);

% 预先分配内存以提高效率
y_chan = zeros(M * N, 1);

for i = 1:num_samples
    % 随机生成目标参数
    h_sample = ones(1, K);

    raw = sort(randsample(4:20-K+1, K, false));
    % 2. 加上偏移量 (核心数学映射)
    % 解释：raw(:) 强转列向量，(0:K-1)' 是列向量
    % 这样相加永远生成列向量，不会变成矩阵
    vals = raw(:) + (0:K-1)';
    % 3. 打乱顺序 (使用索引法)
    % 【关键修复】：这是最安全的打乱方式，避开 randsample 的 bug
    delay_bins_sample = vals(randperm(K));
    raw = sort(randsample(4:20-K+1, K, false));
    % 2. 加上偏移量 (核心数学映射)
    % 解释：raw(:) 强转列向量，(0:K-1)' 是列向量
    % 这样相加永远生成列向量，不会变成矩阵
    vals = raw(:) + (0:K-1)';
    % 3. 打乱顺序 (使用索引法)
    % 【关键修复】：这是最安全的打乱方式，避开 randsample 的 bug
    doppler_bins_sample = vals(randperm(K));
    
    % 添加分数部分，在0到0.2之间取值
    delay_frac_sample = -0.1 + 0.2 * rand(K, 1);
    doppler_frac_sample = -0.1 + 0.2 * rand(K, 1);
    
    delay_sample = delay_bins_sample + delay_frac_sample;
    doppler_sample = doppler_bins_sample + doppler_frac_sample;
    
    delays_sample = delay_sample / (M * Delta_f);
    dopplers_sample = doppler_sample / (N * T);
    
    % 清空信道向量
    y_chan(:) = 0;
    
    % 构造信道矩阵
    for k = 1:K
        y_chan = y_chan + OTFS_output(Xdd, T, delays_sample(k), dopplers_sample(k), SNRdB);
    end

    Ydd = reshape(y_chan, M, N);
    Vdd = computeV_vectorized(Ydd, Xdd);
    % 提取实部和虚部
    real_part = real(Vdd);
    imag_part = imag(Vdd);
    % 使用cat函数沿第一维度拼接成(2, 64, 64)
    % subplot(1,2,1)
    % imagesc(abs(Vdd))
    % colorbar
    % title('幅值 (Magnitude)')
    % axis square
    V_3d = cat(3, real_part, imag_part);
    
    reduced_y = y_chan(kept_indices); % 随机选择保留的行
    
    % 复数转为实数
    [y_real_sample] = complex_to_real_system(reduced_y);
    
    % 计算稀疏表示
    h_sparse2 = Psi' * reduced_y;
    [h_sparse2_sample] = complex_to_real_system(h_sparse2);
    
    % 构建稀疏向量
    h_sparse = zeros(12 * 12, 1);
    % Matlab使用1-based索引
    linear_indices = delay_bins_sample * 12 + doppler_bins_sample + 1;
    h_sparse(linear_indices) = h_sample;
    
    % 检查非零元素数量
    num_nonzero = nnz(h_sparse);
    if num_nonzero ~= K
        fprintf('错误！向量 %d 的非零元素数量不为K\n', i);
        error('非零元素数量不符合要求');
    end
    
    % 构建分数样本
    frac_sample = zeros(12 * 12, 2);
    frac_sample(linear_indices, 1) = delay_frac_sample;
    frac_sample(linear_indices, 2) = doppler_frac_sample;
    
    % 存储样本
    samples{i, 1} = h_sparse;  % 稀疏整数部分
    samples{i, 2} = frac_sample;  % 稀疏分数部分
    samples{i, 3} = y_real_sample;  %接收信号y
    samples{i, 4} = h_sparse2_sample;
    samples{i, 5} = V_3d;  % 二维相关矩阵

    
    % 显示进度
    if mod(i, 100) == 0
        fprintf('  已处理 %d/%d 个数据点 (%.1f%%)\n', i, num_samples, i/num_samples*100);
    end
end

end


function V = computeV_vectorized(Y_DD, X_DD)
    % 矢量化计算V矩阵
    [N, M] = size(Y_DD);
    V = zeros(25, 25);
    
    % 创建索引矩阵
    [k_mat, l_mat] = meshgrid(0:N-1, 0:M-1); % l在行，k在列
    
    % 预计算alpha[k,l]
    alpha = ones(N, M);
    alpha(:, l_mat(1,:) < 0) = exp(-1j * 2 * pi * k_mat(:, l_mat(1,:) < 0) / N);
    
    for k = 0:25-1
        for l = 0:25-1
            % 计算移位索引
            n_indices = mod((0:N-1) - k, N) + 1;
            m_indices = mod((0:M-1) - l, M) + 1;
            
            % 提取并计算乘积
            X_shifted = X_DD(n_indices, m_indices);
            Y_conj = conj(Y_DD);
            exp_term = exp(1j * 2 * pi * ((0:M-1) - l) * k / (N * M));
            
            % 计算V[k,l]
            V(k+1, l+1) = sum(sum(Y_conj .* X_shifted .* alpha(k+1, l+1) .* exp_term)) / (M * N);
        end
    end
end

function F = dft_matrix(N)
% 构造DFT矩阵
% F(k,n) = exp(-2i*pi*(k-1)*(n-1)/N), k,n = 1,2,...,N

k = (0:N-1)';
n = 0:N-1;
F = exp(-2i * pi * k * n / N);
end
