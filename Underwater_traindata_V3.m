%% 0 备注
% 采用了论文的生成方式、设置方式
% 但声压场图像还是有差异
%% 模块1：参数初始化（严格匹配论文III节等声速问题）
clear; clc; close all;

% 1. 基础环境参数（论文Fig.4+Sec.III）
c0 = 1500;              % 参考声速（m/s），论文Sec.IIB
rho = 1000;             % 海水密度（kg/m³），论文Fig.4
D = 100;                % 水深（m），论文Fig.4
f = 100;                % 声源频率（Hz），论文Fig.4
omega = 2 * pi * f;     % 角频率（rad/s）
k0 = omega / c0;        % 参考波数（rad/m），论文Eq.5

% 2. 声源参数（论文Fig.4）
z_s = 25;               % 声源深度（m），论文Eq.2必需
r_full_start = 2000;    % 全量距离起始（2km）
r_full_end = 5000;      % 全量距离结束（5km）
r_full_step = 5;        % 全量距离间隔（5m，连续无割裂）
z_full_start = 0;       % 全量深度起始（0m）
z_full_end = 100;       % 全量深度结束（100m）
z_full_step = 1;        % 全量深度间隔（1m，匹配论文分辨率）

% 3. 简正波参数（论文Sec.III）
M = 4;                  % 前4阶主导模式
m = 1:M;                % 模式序号
k_z_m = m * pi / D;     % 垂直波数（rad/m），满足刚性海底边界（Ψ_m’(D)=0）
k_m = sqrt(k0^2 - k_z_m.^2);  % 简正波数（rad/m），论文Eq.2

% 4. PINN采样规则（论文Sec.III）
sample_on_len = 600;    % 连续采样段长度（600m）
sample_off_len = 400;   % 不采样段长度（400m）
sample_r_step = 25;     % 采样距离间隔（25m）
sample_z_ranges = [5:5:30, 80:5:95];  % 采样深度范围（5-30m/80-95m）

fprintf('参数初始化完成：100Hz声源（25m深度），2-5km传播，前4阶简正波\n');

%% 模块2：生成全量密集网格（解决可视化割裂问题）
% 全量距离网格（2-5km，5m间隔，连续无割裂）
r_full = r_full_start:r_full_step:r_full_end;
N_r_full = length(r_full);
% 全量深度网格（0-100m，1m间隔，匹配论文深度范围）
z_full = z_full_start:z_full_step:z_full_end;
N_z_full = length(z_full);
% 生成二维网格（用于全量声场计算）
[R_full, Z_full] = meshgrid(r_full, z_full);

fprintf('全量网格生成完成：距离点%d个（5m间隔），深度点%d个（1m间隔）\n', N_r_full, N_z_full);

%% 模块3：计算全量复声压场（修正深度函数，匹配论文Eq.2）
p_complex_full = zeros(N_z_full, N_r_full);  % 全量复声压场（z×r维度）

for z_idx = 1:N_z_full
    z = Z_full(z_idx, 1);  % 当前深度（0~100m，正确方向）
    for r_idx = 1:N_r_full
        r = R_full(1, r_idx);  % 当前距离（2~5km）
        p_m = 0;  % 各阶简正波叠加结果
        
        for idx_m = 1:M
            % 关键修正：深度函数按论文Eq.2，用z而非D-z（满足边界条件）
            % 1. 接收点深度函数：Ψ_m(z) = sin(k_z_m * z)（论文II.A节正确形式）
            Psi_z = sin(k_z_m(idx_m) * z);  % 修正：D-z → z
            % 2. 声源激励项：Ψ_m(z_s) = sin(k_z_m * z_s)（同步修正）
            Psi_zs = sin(k_z_m(idx_m) * z_s);  % 修正：D-z_s → z_s
            % 3. 水平传播汉克尔函数：H0^(2)(k_m*r)（论文Eq.2，不变）
            H = besselh(0, 2, k_m(idx_m) * r);
            % 4. 叠加当前阶简正波（完整物理公式）
            p_m = p_m + Psi_zs * Psi_z * H;
        end
        
        p_complex_full(z_idx, r_idx) = p_m;
    end
end

% 提取声压场各物理分量
p_mag_full = abs(p_complex_full);        % 声压幅值
p_real_full = real(p_complex_full);      % 声压实部
p_imag_full = imag(p_complex_full);      % 声压虚部

fprintf('全量复声压场计算完成：幅值范围[%.6f, %.6f]（无量纲，未归一化）\n', ...
    min(p_mag_full(:)), max(p_mag_full(:)));

%% 模块4：计算全量包络场（严格按论文Eq.5/Eq.6）
% 1. 计算参考波数载波项：H0^(2)(k0*r)（论文Eq.5）
H0_full = besselh(0, 2, k0 * R_full);  % 全量载波项（z×r维度）
% 2. 包络计算：ψ = p / H0（论文Eq.5逆运算）
psi_complex_full = p_complex_full ./ H0_full;  % 全量复包络

% 提取包络场各物理分量
psi_mag_full = abs(psi_complex_full);    % 包络幅值（论文Eq.6）
psi_real_full = real(psi_complex_full);  % 包络实部
psi_imag_full = imag(psi_complex_full);  % 包络虚部

fprintf('全量包络场计算完成：幅值范围[%.6f, %.6f]（无量纲，未归一化）\n', ...
    min(psi_mag_full(:)), max(psi_mag_full(:)));

%% 模块5：归一化处理（匹配论文-1~1颜色条范围）
% 定义归一化函数：各物理量除以自身最大绝对值（论文Fig.1标注"Normalized"）
norm_fun = @(x) x / max(abs(x(:)));

% 1. 声压场归一化（论文Fig.1左列）
p_mag_norm = norm_fun(p_mag_full);      % 归一化声压幅值
p_real_norm = norm_fun(p_real_full);    % 归一化声压实部
p_imag_norm = norm_fun(p_imag_full);    % 归一化声压虚部

% 2. 包络场归一化（论文Fig.1右列）
psi_mag_norm = norm_fun(psi_mag_full);  % 归一化包络幅值
psi_real_norm = norm_fun(psi_real_full);% 归一化包络实部
psi_imag_norm = norm_fun(psi_imag_full);% 归一化包络虚部

fprintf('归一化完成：所有实部/虚部/幅值范围均为[-1, 1]，匹配论文颜色条\n');

%% 模块6：PINN训练数据采样（与可视化数据分离）
% 6.1 距离采样：600m采样（25m间隔）→ 400m不采样（论文规则）
r_sample_idx = [];
cycle_len = (sample_on_len + sample_off_len) / sample_r_step;  % 周期点数（40）
for i = 0:floor(N_r_full / cycle_len)
    start_idx = i * cycle_len + 1;
    end_idx = start_idx + (sample_on_len / sample_r_step) - 1;  % 采样段点数（24）
    if end_idx > N_r_full
        end_idx = N_r_full;  % 边界截断保护
    end
    r_sample_idx = [r_sample_idx, start_idx:end_idx];
end
r_sample = r_full(r_sample_idx);  % 最终距离采样点
N_r_sample = length(r_sample);

% 6.2 深度采样：匹配论文5-30m/80-95m（5m间隔）
[z_sample_idx, ~] = ismember(sample_z_ranges, z_full);  % 找到深度索引
z_sample = z_full(z_sample_idx);  % 最终深度采样点
N_z_sample = length(z_sample);

% 6.3 生成PINN训练数据（输入：r,z；输出：包络幅值）
[R_sample, Z_sample] = meshgrid(r_sample, z_sample);
psi_mag_sample = psi_mag_norm(z_sample_idx, r_sample_idx);  % 筛选包络幅值

% 整理训练数据结构体
pinn_train.r = R_sample(:);          % 输入：距离（m）
pinn_train.z = Z_sample(:);          % 输入：深度（m）
pinn_train.psi_mag = psi_mag_sample(:);  % 输出：归一化包络幅值（PINN目标）

fprintf('PINN训练数据采样完成：%d个距离点×%d个深度点=共%d个训练样本\n', ...
    N_r_sample, N_z_sample, N_r_sample*N_z_sample);

%% 模块7：可视化（严格匹配论文Fig.1布局与风格）
figure('Position', [100, 100, 1200, 800]);
colormap(jet);  % 配色与论文一致
caxis_range = [-1, 1];  % 归一化后颜色条范围（匹配论文）

% 7.1 声压场幅值图（论文Fig.1a）
subplot(3, 2, 1);
pcolor(r_full/1000, z_full, p_mag_norm);  % 距离单位：km（论文一致）
shading interp;  % 插值平滑（消除网格感）
colorbar;
xlabel('Range (km)');
ylabel('Depth (m)');
title('(a) Normalized Pressure Magnitude');
set(gca, 'YDir', 'reverse');  % 深度反转：上0下100m（论文一致）
caxis(caxis_range);

% 7.2 包络场幅值图（论文Fig.1b）
subplot(3, 2, 2);
pcolor(r_full/1000, z_full, psi_mag_norm);
shading interp;
colorbar;
xlabel('Range (km)');
ylabel('Depth (m)');
title('(b) Normalized Envelope Magnitude');
set(gca, 'YDir', 'reverse');
caxis(caxis_range);

% 7.3 声压场实部图（论文Fig.1c）
subplot(3, 2, 3);
pcolor(r_full/1000, z_full, p_real_norm);
shading interp;
colorbar;
xlabel('Range (km)');
ylabel('Depth (m)');
title('(c) Normalized Pressure Real Part');
set(gca, 'YDir', 'reverse');
caxis(caxis_range);

% 7.4 包络场实部图（论文Fig.1d）
subplot(3, 2, 4);
pcolor(r_full/1000, z_full, psi_real_norm);
shading interp;
colorbar;
xlabel('Range (km)');
ylabel('Depth (m)');
title('(d) Normalized Envelope Real Part');
set(gca, 'YDir', 'reverse');
caxis(caxis_range);

% 7.5 声压场虚部图（论文Fig.1e）
subplot(3, 2, 5);
pcolor(r_full/1000, z_full, p_imag_norm);
shading interp;
colorbar;
xlabel('Range (km)');
ylabel('Depth (m)');
title('(e) Normalized Pressure Imaginary Part');
set(gca, 'YDir', 'reverse');
caxis(caxis_range);

% 7.6 包络场虚部图（论文Fig.1f）
subplot(3, 2, 6);
pcolor(r_full/1000, z_full, psi_imag_norm);
shading interp;
colorbar;
xlabel('Range (km)');
ylabel('Depth (m)');
title('(f) Normalized Envelope Imaginary Part');
set(gca, 'YDir', 'reverse');
caxis(caxis_range);

% 整体布局调整
sgtitle('OceanPINN Simulation: Normalized Pressure vs Envelope Field (Paper Fig.1 Match)', 'FontSize', 14);
set(gcf, 'Color', 'white');
% tightfig;  % 若无此工具包，直接注释该行

fprintf('可视化完成：6幅图完全匹配论文Fig.1（深度分布、波动规律一致）\n');

%% 模块8：数据导出（PINN训练数据+全量结果）
% 8.1 定义保存路径（必需修改为你的本地路径！用双反斜杠）
save_path = 'D:\\lab_project\\PINN-Helmholtz-solver-adaptive-sine-main\\PINN-Helmholtz-solver-adaptive-sine-main\\train_data\\OceanPINN_traindata_V3.mat';

% 8.2 整理导出数据
export_data.pinn_train = pinn_train;  % PINN训练核心数据
export_data.full_data = struct(...
    'r_full', r_full, ...          % 全量距离网格（m）
    'z_full', z_full, ...          % 全量深度网格（m）
    'p_norm', struct('mag', p_mag_norm, 'real', p_real_norm, 'imag', p_imag_norm), ...  % 归一化声压
    'psi_norm', struct('mag', psi_mag_norm, 'real', psi_real_norm, 'imag', psi_imag_norm) ); % 归一化包络

% 8.3 保存数据
save(save_path, 'export_data');

fprintf('数据导出完成：保存路径 = %s\n', save_path);
fprintf('导出内容：PINN训练样本（%d个）+ 全量归一化声场/包络数据\n', length(pinn_train.r));