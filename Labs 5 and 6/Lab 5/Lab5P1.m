% BIOMEDE 517 - Neural Engineering
% Lab 5 Part 1
% Kushal Jaligama

% Part 1
% In this part, we use data from Williams 2007
% The frequencies of the points on the plot are taken from 100 Hz to 2 kHz
% in 100 Hz increments

% Estimate Ren + Rex - extrapolate the first few points
% Point = [Z_real, Z_imaginary]
Blue1 = [175000, 250000]
Blue2 = [110000, 160000]
slopeBlue = (Blue1(2) - Blue1(2)) / (Blue2(1) - Blue2(1));
blue_x_intercept = (slopeBlue * Blue1(1) - Blue1(2)) / slopeBlue;
RenplusRexBlue = blue_x_intercept; % Ohms
ZreBlue = Blue1(1); % Ohms
ZimBlue = Blue1(2); % Ohms
% Calculate the alpha value
alphaBlue = 2/pi()*atan(slopeBlue);

Red1 = [420000, 250000]
Red2 = [375000, 175000]
slopeRed = (Red1(2) - Red2(2)) / (Red1(1) - Red2(1));
red_x_intercept = (slopeRed * Red1(1) - Red1(2)) / slopeRed;
RenplusRexRed = red_x_intercept; % Ohms
ZreRed = Red1(1); % Ohms
ZimRed = Red1(2); % Ohms
% Calculate the alpha value
alphaRed = 2 / pi() * atan(slopeRed);

% By looking at values, RenplusRexBlue is NaN so make it zero
RenplusRexBlue = 0;

% In the paper the points range from 2Khz to 100Hz in increments of 100
% Z_real and Z_re for calculating K were gathered at 100 Hz
w_naught = j * 2 * pi() * 100;

% Calculate K for both lines
% Magnitude of a complex number (a + bi) is sqrt(a^2 + b^2)
% Magnitude of a complex number is also the absolute value of it
% K = (Zre(w_naught) - jZim(w_naught) - (Ren+Rex)) * (j*w_naught)^alpha
% The parenthesis in Zre(w_naught) are NOT multiplication, rather functions
% to grab the Zre at the 0th frequency (which is 100 in this paper).
KBlue = (ZreBlue - j*ZimBlue - RenplusRexBlue) * (j*w_naught)^alphaBlue
KRed = (ZreRed - j*ZimRed - RenplusRexRed) * (j*w_naught)^alphaRed

% Calculate magnitude of tissue related response (Ztotal) at 1000 Hz
w = j * 2 * pi() * 1000;
ZtotalBlue = abs(RenplusRexBlue + KBlue / (j * w)^alphaBlue)
ZtotalRed = abs(RenplusRexRed + KRed / (j * w)^alphaRed)

% ZtotalBlue and ZtotalRed are of the form (a + bi)

% What conclusions can you make about the electrode performance and the tissue response at the time of implantation and seven days later?
