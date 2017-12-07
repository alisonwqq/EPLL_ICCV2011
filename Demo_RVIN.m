
clear
patchSize = 8;

Original_image_dir  =    'C:\Users\csjunxu\Desktop\ECCV2016\grayimages\';
fpath = fullfile(Original_image_dir, '*.png');
im_dir  = dir(fpath);
im_num = length(im_dir);
for nSig = [10 20]
    for SpikyRatio = [0.15 0.3]
        % load image
        PSNR = [];
        SSIM = [];
        for i = 1:im_num
            I = single( imread(fullfile(Original_image_dir, im_dir(i).name)) )/255;
            %% add Gaussian noise
            randn('seed',0);
            noiseI = I + nSig/255*randn(size(I));
            %% add spiky noise or "salt and pepper" noise 1
            %                 rand('seed',Sample-1)
            %                 N_Img = 255*imnoise(N_Img/255, 'salt & pepper', SpikyRatio); %"salt and pepper" noise
            %% add spiky noise or "salt and pepper" noise 2
            rand('seed',0)
            [noiseI,Narr]          =   impulsenoise(noiseI*255,SpikyRatio,1);
            fprintf('The initial value of PSNR = %2.2f  SSIM=%2.4f\n', csnr( noiseI, I*255 , 0, 0 ), cal_ssim(noiseI, I*255 , 0, 0 ));
            %% AMF
            [noiseIAMF,ind]=adpmedft(noiseI,19);
            %% noise level estimation
            nLevel = NoiseLevel(noiseIAMF);
            excludeList = [];
            % set up prior
            LogLFunc = [];
            load GSModel_8x8_200_2M_noDC_zeromean.mat
            prior = @(Z,patchSize,noiseSD,imsize) aprxMAPGMM(Z,patchSize,nLevel,imsize,GS,excludeList);
            
            %%
            tic
            %% add 64 and 128 for high noise
            [cleanI,psnr,~] = EPLLhalfQuadraticSplit(noiseIAMF/255,patchSize^2/nLevel^2,patchSize,(1/nLevel^2)*[1 4 8 16 32 64],1,prior,I,LogLFunc);
            toc
            
            % output result
            PSNR =  [PSNR csnr( cleanI*255, I*255, 0, 0 )];
            SSIM      =  [SSIM cal_ssim( cleanI*255, I*255, 0, 0 )];
            imname = sprintf('./GauRVIN/EPLL_AMF_GauRVIN_%d_%2.2f_%s',nSig,SpikyRatio,im_dir(i).name);
            imwrite(cleanI,imname);
            fprintf('%s : PSNR = %2.4f, SSIM = %2.4f \n',im_dir(i).name,csnr( cleanI*255, I*255, 0, 0 ),cal_ssim( cleanI*255, I*255, 0, 0 ));
        end
        mPSNR = mean(PSNR);
        mSSIM = mean(SSIM);
        name = sprintf('EPLL_AMF_GauRVIN_%d_%2.2f.mat',nSig,SpikyRatio);
        save(name,'mPSNR','PSNR','SSIM','mSSIM');
    end
end