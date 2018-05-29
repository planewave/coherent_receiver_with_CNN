% generate passband QPSK signal for DL 
% for decoder training with GPU
% with random SNR
clear
close all
clc
% Set the simulation parameters.

M = 4;                                   % Modulation order
k = log2(M);                             % Bits/symbol
nSym = 128;                              % Transmitted symbols
NUM_WAV = 1;
batch = 1024;
% EbNo = 15;                               % Eb/No (dB)
span = 6;                                % Filter span in symbols
rolloff = 0.4;                           % Rolloff factor
fc = 4.33e3;                               % carrier freq
% fs = 3.2*fc;
fsym = 1e3;                              % symbol rate
sps = 16;
fs = sps*fsym;

% offset = int64(randi([0, 15], [batch, 1]));
offset = int64(randi([4, 11], [batch, 1]));

x = zeros(batch, 1, sps+4, nSym);
for step = 1:batch
    
    msg = randi([0 M-1],nSym,NUM_WAV);
    modData = pskmod(msg, M, pi/4);
    modDataUp = upsample(modData,sps);
    % pulse shaping
    rrc = rcosdesign(rolloff,span,sps);
    txPulse = conv(modDataUp, rrc)*sqrt(sps);
    t = (0:1/fs:(length(txPulse)-1)/fs)';
    
    % txPulse = upfirdn(modData, rrcFilter, sps)*sqrt(sps);
    % t = (0:1/fs:(nSym+span-1)/fsym)';
    phi = 0; % initial phase, 0 for now
    carrier = 1.414*exp(1j*(2*pi*fc*t+phi))*ones(1, NUM_WAV);
    
    txPass = real(txPulse.*carrier);
    
    
    txPass = circshift(txPass,offset(step));
    
    % offset = -0;
    % txPass = circshift(txPulse,offset);
    % plotEye(txPass(49:end-49),sps)
    
    
    EbN0 = 10;
    SNR = EbN0 + 10*log10(k) - 10*log10(sps);
    % noisySig = awgn(txPass,SNR,'measured');
    % msg = int64(msg(2:nSym-1, :)); % for LongTensor
    noisySig = awgn(txPass, SNR+3,'measured');
    
    filtDelay = sps*(span)/2;
    
    noisysigShap = zeros(sps+4,nSym);
    for symb = 1:nSym
        start = (symb-1)*sps-1+filtDelay;
        noisysigShap(:, symb) = noisySig(start:start+sps+3);
    end
    
    
%     noisysigShap = reshape(noisySig(filtDelay+1:filtDelay+sps*nSym), sps, nSym);
    x(step, 1, :, :) = noisysigShap;
end
x = single(x);  % for FloatTensor
save('sto10dBfc433.mat', 'x', 'offset');



% rxDown = noisySig.*conj(carrier);
% rxFilt = conv(rxDown, rrc);
% plotEye(rxFilt(89:end-89),sps)
% rxFilt = upfirdn(rxDown, rrc, 1, 1); % receiver matched filter
% dataSet = [abs(rxFilt.'), offset];
% dataSet2 = [abs(rxFilt.').^2, offset];
%%
% txPass = txPass(41:end-41);
% rxDown = txPass.*conj(carrier(41:end-41));
% rxFilt = conv(rxDown, rrc);
% rxFilt = rxFilt(48+1:end-48);
% eyediagram(txPulse(41:end-41),sps)


% testSig = rxFilt(:, 1);
% testSig = testSig(span*sps+1:end-span*sps);
% eyediagram(testSig,sps)

rxDwonSamp = rxFilt(97:sps:end-97); % down sampling
% rxDwonSamp = rxDwonSamp(span+1:end-span); % remove filter introduced delay

rxdata = pskdemod(rxDwonSamp, M, pi/4);
accuracy = sum(msg == rxdata)/nSym*100


% sig = single(noisySig);
% txPass = single(noisySig(81+off:32*nSym+80+off,:));  % for FloatTensor
% save(['train', num2str(EbN0), 'dB.mat'], 'sig', 'msg','NUM_WAV','EbN0');

% data = int64(data); % for LongTensor
% txPass = single(noisySig);  % for FloatTensor
% save('trainPerfect10dB.mat', 'data', 'txPass','NUM_WAV');
%% test data
close all
nSym = 200;
% fc = 10e3;                               % carrier freq with offset
NUM_WAV = 1;
M = 4;
msg = randi([0 M-1],nSym,1);
modData = pskmod(msg, M, pi/4);

% M = 2;
% msg = randi([0 M-1],nSym,1);
% modData = msg*2-1;
var

EbN0 = 30;                               % Eb/N0 (dB)
% pulse shapingccc
rrc = rcosdesign(rolloff,span,sps);
txPulse = upfirdn(modData, rrc, sps);
t = (0:1/fs:(nSym+span-1)/fsym)';
phi = 0; % initial phase 
carrier = exp(1j*(2*pi*fc*t+phi));
txPass = real(txPulse.*carrier);
% txPass = txPass+0.6*circshift(txPass, round(sps*1.4))+0.3*circshift(txPass, round(sps*3.5));

SNR = EbN0 + 10*log10(k) - 10*log10(sps);
    % For passband signal, its bandwidth is 2 times of its
    % baseband complex envelop, so to have the same noise energy
    % we need +3dB SNR
noisySig = awgn(txPass,SNR+3,'measured');
msg = int64(msg); % for LongTensor
start = sps*2.5+1; %span = 6;
sig = single(noisySig(start:end-start));



% save(['test', num2str(EbN0), 'dB.mat'], 'msg', 'sig','nSym');

rxDown = noisySig.*conj(carrier);
% rxDown = circshift(rxDown, off);
% rxFilt = upfirdn(rxDown, rrcFilter, 1, sps);

rxFilt = conv(rrc, rxDown); % receiver matched filter

tauOm = round(-angle(sum(abs(rxFilt).^2.*exp(-1j*2*pi*(0:length(rxFilt)-1)'/sps)))/2/pi*sps);


rxDwonSamp = rxFilt(1:sps:end); % down sampling

rxDwonSamp = rxDwonSamp(span+1:end-span); % remove filter introduced delay
% scatterplot(rxFilt)
rxdata = pskdemod(rxDwonSamp, M, pi/4);
accuracy = sum(msg == rxdata)/nSym*100;
errate = sum(msg ~= rxdata)/nSym
% check eye
eyediagram(abs(rxFilt(sps*10+1:1000)).^2,sps)
eyediagram((rxFilt(sps*10+1:1500)),sps)

%% only for test
% sps = 100;
% fsym = 1e3;                              % symbol rate
% fs = sps*fsym;
% modData = pskmod(3, 4, pi/4);
% rrcFilter = rcosdesign(rolloff,span,sps);
% txSig = upfirdn(modData, rrcFilter, sps);
% t = (0:1/fs:span/fsym);
% t = 200/fs:1/fs:(span+2)/fsym;
% fc = 6e3;                               % carrier freq
% carrier = exp(1j*(2*pi*fc*t));
% txPass = sqrt(2)*real(txSig.*carrier);
% 
% fc2 = 10e3+2;                               % carrier freq
% carrier2 = exp(1j*(2*pi*fc2*t));
% txPass2 = sqrt(2)*real(txSig.*carrier2); 
% 
% plot(txPass)
% hold