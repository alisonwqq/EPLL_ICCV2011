clear
addpath('NoiseEstimation');
GT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2018 Denoising\cc_Results\Real_ccnoise_denoised_part\';
GT_fpath = fullfile(GT_Original_image_dir, '*mean.png');
TT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2018 Denoising\cc_Results\Real_ccnoise_denoised_part\';
TT_fpath = fullfile(TT_Original_image_dir, '*real.png');
% GT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2018 Denoising\cc_Results\Real_MeanImage\';
% GT_fpath = fullfile(GT_Original_image_dir, '*.png');
% TT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2018 Denoising\cc_Results\Real_NoisyImage\';
% TT_fpath = fullfile(TT_Original_image_dir, '*.png');
% GT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2018 Denoising\our_Results\Real_MeanImage\';
% GT_fpath = fullfile(GT_Original_image_dir, '*.JPG');
% TT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2018 Denoising\our_Results\Real_NoisyImage\';
% TT_fpath = fullfile(TT_Original_image_dir, '*.JPG');
% GT_Original_image_dir = 'C:/Users/csjunxu/Desktop/RID_Dataset/RealisticImage/';
% GT_fpath = fullfile(GT_Original_image_dir, '*mean.JPG');
% TT_Original_image_dir = 'C:/Users/csjunxu/Desktop/RID_Dataset/RealisticImage/';
% TT_fpath = fullfile(TT_Original_image_dir, '*real.JPG');
GT_im_dir  = dir(GT_fpath);
TT_im_dir  = dir(TT_fpath);
im_num = length(TT_im_dir);

method = 'EPLL';
dataset = 'cc';
write_MAT_dir = ['C:/Users/csjunxu/Desktop/CVPR2018 Denoising/cc_Results/'];
write_sRGB_dir = [write_MAT_dir method];
if ~isdir(write_sRGB_dir)
    mkdir(write_sRGB_dir)
end
patchSize = 8;
PSNR = [];
SSIM = [];
RunTime = [];
for i = 1 : im_num
    IM =   double(imread( fullfile(TT_Original_image_dir,TT_im_dir(i).name) ));
    IM_GT = double(imread(fullfile(GT_Original_image_dir, GT_im_dir(i).name)));
    fprintf('The initial PSNR = %2.4f, SSIM = %2.4f. \n', csnr(uint8(IM), uint8(IM_GT), 0, 0 ), cal_ssim(uint8(IM), uint8(IM_GT), 0, 0 ));
    % S = regexp(TT_im_dir(i).name, '\.', 'split');
    IMname = TT_im_dir(i).name(1:end-9);
    [h,w,ch] = size(IM);
    excludeList = [];
    
    % set up prior
    LogLFunc = [];
    load GSModel_8x8_200_2M_noDC_zeromean.mat
    prior = @(Z,patchSize,noiseSD,imsize) aprxMAPGMM(Z,patchSize,noiseSD,imsize,GS,excludeList);
    
    time0 = clock;
    IMout = zeros(size(IM));
    for cc = 1:ch
        noiseI = IM(:, :, cc);
        I = IM_GT(:, :, cc);
        %% noise estimation %%
        noiseSD = NoiseEstimation(noiseI, 8)/255;
        %% denoising
        % add 64 and 128 for high noise
        [IMoutcc,~,~] = EPLLhalfQuadraticSplit(noiseI/255,patchSize^2/noiseSD^2,patchSize,(1/noiseSD^2)*[1 4 8 16 32 64],1,prior,I/255,LogLFunc);
        IMout(:,:,cc) = IMoutcc*255;
    end
    RunTime = [RunTime etime(clock,time0)];
    fprintf('Total elapsed time = %f s\n', (etime(clock,time0)) );
    PSNR = [PSNR csnr( uint8(IMout), uint8(IM_GT), 0, 0 )];
    SSIM = [SSIM cal_ssim( uint8(IMout), uint8(IM_GT), 0, 0 )];
    fprintf('The final PSNR = %2.4f, SSIM = %2.4f. \n', PSNR(end), SSIM(end));
    imwrite(IMout/255, [write_sRGB_dir '/' method '_' dataset '_' IMname '.png']);
end
mPSNR = mean(PSNR);
mSSIM = mean(SSIM);
mRunTime = mean(RunTime);
matname = sprintf([write_MAT_dir method '_' dataset '.mat']);
save(matname,'PSNR','mPSNR','SSIM','mSSIM','RunTime','mRunTime');
