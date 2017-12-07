clear
addpath('NoiseEstimation');
Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2018 Denoising\dnd_2017\images_srgb\';
fpath = fullfile(Original_image_dir, '*.mat');
im_dir  = dir(fpath);
im_num = length(im_dir);
load 'C:\Users\csjunxu\Desktop\CVPR2018 Denoising\dnd_2017\info.mat';

method = 'EPLL';
dataset = 'dnd_2017';
write_MAT_dir = ['C:/Users/csjunxu/Desktop/CVPR2018 Denoising/' dataset '_Results/'];
write_sRGB_dir = [write_MAT_dir method];
if ~isdir(write_sRGB_dir)
    mkdir(write_sRGB_dir)
end


patchSize = 8;

RunTime = [];
for i = 1 :im_num
    load(fullfile(Original_image_dir, im_dir(i).name));
    S = regexp(im_dir(i).name, '\.', 'split');
    [h,w,ch] = size(InoisySRGB);
    for j = 1:size(info(1).boundingboxes,1)
        IMinname = [S{1} '_' num2str(j)];
        bb = info(i).boundingboxes(j,:);
        nim = InoisySRGB(bb(1):bb(3), bb(2):bb(4),:)*255;
        fprintf('%s: \n', IMinname);
        % noise estimation
        t1=clock;
        excludeList = [];
        % set up prior
        LogLFunc = [];
        load GSModel_8x8_200_2M_noDC_zeromean.mat
        prior = @(Z,patchSize,noiseSD,imsize) aprxMAPGMM(Z,patchSize,noiseSD,imsize,GS,excludeList);
        
        IMout = zeros(size(nim));
        for c = 1:ch
            noiseI = nim(:, :, c);
            I = noiseI;
            noiseSD = NoiseEstimation(noiseI, 8)/255;
            %% denoising
            % add 64 and 128 for high noise
            [IMoutc,~,~] = EPLLhalfQuadraticSplit(noiseI/255,patchSize^2/noiseSD^2,patchSize,(1/noiseSD^2)*[1 4 8 16 32 64],1,prior,I/255,LogLFunc);
            IMout(:,:,c) = IMoutc;
        end
        t2=clock;
        etime(t2,t1)
        alltime(i)  = etime(t2, t1);
        %% output
        IMoutname = sprintf([write_sRGB_dir '/' method '_DND_' IMinname '.png']);
        imwrite(IMout, IMoutname);
    end
end