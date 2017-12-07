clear
patchSize = 8;

Original_image_dir  =    './Ground_Truth/';
fpath = fullfile(Original_image_dir, '*.png');
im_dir  = dir(fpath);
im_num = length(im_dir);
for nSig = [40 50 60 75 100]
    % load image
    PSNR = [];
    SSIM = [];
    for i = 1:im_num
        noiseSD = nSig/255;
        I = single( imread(fullfile(Original_image_dir, im_dir(i).name)) )/255;
        
        % add noise
        randn('seed',0);
        noiseI = I + noiseSD*randn(size(I));
        excludeList = [];
        
        % set up prior
        LogLFunc = [];
        load GSModel_8x8_200_2M_noDC_zeromean.mat
        prior = @(Z,patchSize,noiseSD,imsize) aprxMAPGMM(Z,patchSize,noiseSD,imsize,GS,excludeList);
        
        %%
        tic
        % add 64 and 128 for high noise
        [cleanI,psnr,~] = EPLLhalfQuadraticSplit(noiseI,patchSize^2/noiseSD^2,patchSize,(1/noiseSD^2)*[1 4 8 16 32 64],1,prior,I,LogLFunc);
        toc
        
        % output result
        figure(1);
        imshow(I); title('Original');
        figure(2);
        imshow(noiseI); title('Corrupted Image');
        figure(3);
        imshow(cleanI); title('Restored Image');
        PSNR =  [PSNR csnr( cleanI*255, I*255, 0, 0 )];
        SSIM      =  [SSIM cal_ssim( cleanI*255, I*255, 0, 0 )];
        imname = sprintf('EPLL_nSig%d_%s',nSig,im_dir(i).name);
        imwrite(cleanI,imname);
        fprintf('%s : PSNR = %2.4f, SSIM = %2.4f \n',im_dir(i).name,csnr( cleanI*255, I*255, 0, 0 ),cal_ssim( cleanI*255, I*255, 0, 0 ));
    end
    mPSNR = mean(PSNR);
    mSSIM = mean(SSIM);
    name = sprintf('EPLL_nSig%d.mat',nSig);
    save(name,'mPSNR','PSNR','SSIM','mSSIM');
end
