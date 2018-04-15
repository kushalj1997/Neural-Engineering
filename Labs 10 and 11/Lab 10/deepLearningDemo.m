%%Deep learning demonstration using AlexNet and webcam

%NOTE: requires some setup using the add-on explorer
% - need to use add-ons explorer to install webcam support package
% - need to use add-ons explorer to install Neural Network Toolbox Model for AlexNet Network

clear camera;
camera = webcam; %variable for webcam
net = alexnet; %load in AlexNet neural network
inputSize = net.Layers(1).InputSize(1:2); %size of images AlexNet wants

%Initialize figure
h = figure;
h.Position(3) = 2*h.Position(3);
ax1 = subplot(1,2,1);
ax2 = subplot(1,2,2);
ax2.ActivePositionProperty = 'position';

%Keep this running in a loop until the figure is closed
keepRolling = true;
set(gcf,'CloseRequestFcn','keepRolling = false; closereq');

while keepRolling
    % Display and classify the image
    im = snapshot(camera);
    image(ax1,im)
    im = imresize(im,inputSize); %resize webcam image
    [label,score] = classify(net,im);
    title(ax1,{char(label),num2str(max(score),2)});

    % Select the top five predictions
    [~,idx] = sort(score,'descend');
    idx = idx(5:-1:1);
    scoreTop = score(idx);
    classNames = net.Layers(end).ClassNames;
    classNamesTop = classNames(idx);

    % Plot the histogram
    barh(ax2,scoreTop)
    title(ax2,'Top 5')
    xlabel(ax2,'Probability')
    xlim(ax2,[0 1])
    yticklabels(ax2,classNamesTop)
    ax2.YAxisLocation = 'right';

    drawnow %update figures
end