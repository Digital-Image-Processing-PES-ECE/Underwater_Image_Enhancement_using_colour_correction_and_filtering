function uiqm = calculate_uiqm(image)
    img = im2double(image);
    R = img(:,:,1);
    G = img(:,:,2);
    B = img(:,:,3);

    rg = R - G;
    yb = 0.5 * (R + G) - B;
    uicm = sqrt(mean(rg(:).^2) + mean(yb(:).^2)); 

    sobel_response = abs(imfilter(rgb2gray(img), fspecial('sobel'), 'replicate'));
    uism = mean(sobel_response(:)); 

    intensity = rgb2gray(img);
    local_std = stdfilt(intensity, ones(3)); 
    uiconm = mean(local_std(:));

    c1 = 0.5; 
    c2 = 0.3; 
    c3 = 0.2; 
    uiqm = c1 * uicm + c2 * uism + c3 * uiconm;
end

% PSNR Calculation Function
function psnr_value = calculate_psnr(original, processed)
    mse = mean((double(original(:)) - double(processed(:))).^2);
    psnr_value = 10 * log10(255^2 / mse);
end

img = imread("C:\Users\praga\OneDrive\Desktop\SEM 5\DIP Lab\DIP-OpenCV-Learning\dip-venv\Project\InputImages\116_img_.png");

original_psnr_reference = double(img);

%% 1. White Balance Calculation
img_double = double(img);

mean_R = mean(mean(img_double(:,:,1)));
mean_G = mean(mean(img_double(:,:,2)));
mean_B = mean(mean(img_double(:,:,3)));

scale_R = mean_G / mean_R;
scale_B = mean_G / mean_B;

img_double(:,:,1) = img_double(:,:,1) * scale_R;
img_double(:,:,3) = img_double(:,:,3) * scale_B;

white_balanced_img = uint8(img_double);

figure;
subplot(1,2,1)
imshow(img);
title('Original Image');
subplot(1,2,2)
imshow(white_balanced_img);
title('White Balanced Image');

x = calculate_uiqm(white_balanced_img);
psnr_white_balanced = calculate_psnr(original_psnr_reference, white_balanced_img);
disp('UIQM for White Balanced Image:');
disp(x);
disp('PSNR for White Balanced Image:');
disp(psnr_white_balanced);

%% 2. High Contrast and Edge Sharpening
R = white_balanced_img(:,:,1);
G = white_balanced_img(:,:,2);
B = white_balanced_img(:,:,3);

E_R = edge(R, "sobel");
E_G = edge(G, "sobel");
E_B = edge(B, "sobel");

sharp_R = uint8(E_R) + R;
sharp_G = uint8(E_G) + G;
sharp_B = uint8(E_B) + B;

eq_R = histeq(sharp_R);
eq_G = histeq(sharp_G);
eq_B = histeq(sharp_B);

R_filtered = wiener2(eq_R, [5 5]);
G_filtered = wiener2(eq_G, [5 5]);
B_filtered = wiener2(eq_B, [5 5]);

final_img = cat(3, R_filtered, G_filtered, B_filtered);

figure;
subplot(1,2,1)
imshow(white_balanced_img);
title('White Balanced Image');
subplot(1,2,2)
imshow(final_img);
title('High Contrast and Edge Sharpened');

x = calculate_uiqm(final_img);
psnr_final = calculate_psnr(double(white_balanced_img), final_img);
disp('UIQM for High Contrast and Edge Sharpened Image:');
disp(x);
disp('PSNR for High Contrast and Edge Sharpened Image:');
disp(psnr_final);

%% 3. Dehazing or Contrast Enhancement
lab_img = rgb2lab(white_balanced_img);
L = lab_img(:,:,1);
L = histeq(L / 100) * 100;
lab_img(:,:,1) = L;

dehazed_img = lab2rgb(lab_img);

figure;
subplot(1,2,1)
imshow(white_balanced_img);
title('White Balanced Image');
subplot(1,2,2)
imshow(dehazed_img);
title('Dehazed Image');

x = calculate_uiqm(dehazed_img);
psnr_dehazed = calculate_psnr(double(white_balanced_img), dehazed_img);
disp('UIQM for Dehazed Image:');
disp(x);
disp('PSNR for Dehazed Image:');
disp(psnr_dehazed);

%% 4. DCP Enhancement
lab_img = rgb2lab(white_balanced_img);
L = lab_img(:,:,1);
L_eq = adapthisteq(L / 100) * 100;
lab_img(:,:,1) = L_eq;

dcp_img = lab2rgb(lab_img);

figure;
subplot(1,2,1)
imshow(white_balanced_img);
title('White Balanced Image');
subplot(1,2,2)
imshow(dcp_img);
title('DCP Enhanced Image');

x = calculate_uiqm(dcp_img);
psnr_dcp = calculate_psnr(double(white_balanced_img), dcp_img);
disp('UIQM for DCP Enhanced Image:');
disp(x);
disp('PSNR for DCP Enhanced Image:');
disp(psnr_dcp);

%% 5. Red Channel Compensation
img_double = double(white_balanced_img);
red_boost_factor = 1.5;
img_double(:,:,1) = img_double(:,:,1) * red_boost_factor;
img_double(:,:,1) = min(img_double(:,:,1), 255);
red_compensated_img = uint8(img_double);

figure;
subplot(1,2,1)
imshow(white_balanced_img);
title('White Balanced Image');
subplot(1,2,2)
imshow(red_compensated_img);
title('Red Channel Compensated Image');

x = calculate_uiqm(red_compensated_img);
psnr_red_compensated = calculate_psnr(double(white_balanced_img), red_compensated_img);
disp('UIQM for Red Channel Compensated Image:');
disp(x);
disp('PSNR for Red Channel Compensated Image:');
disp(psnr_red_compensated);
