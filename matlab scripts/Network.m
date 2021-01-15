% 경로 및 읽어올 파일, 결과 저장 파일 설정
folder_path = "";
C = readtable(folder_path+"network.csv")
result_file = folder_path+"model_sync_Matlab_result.xlsx";

% 데이터 필터링
ps_tx_data = table2array(C(:,2));
ps_rx_data = table2array(C(:,3));
w1_tx_data = table2array(C(:,4));
w1_rx_data = table2array(C(:,5));
w2_tx_data = table2array(C(:,6));
w2_rx_data = table2array(C(:,7));

% 함수 / 데이터 종류 6개 (PS, W1, W2 & 각각 tx, rx)
% net_func(data, gpu_num, data_type, peak_threshold, start_rm_peak, end_rm_peak)
T = net_func(ps_tx_data, "PS", "tx", 0, 1, 0);
writematrix(T,result_file,'Sheet',"Network",'Range','B2');
T = net_func(ps_rx_data, "PS", "rx", 0, 1, 0);
writematrix(T,result_file,'Sheet',"Network",'Range','E2');
T = net_func(w1_tx_data, "W1", "tx", 0, 1, 0);
writematrix(T,result_file,'Sheet',"Network",'Range','H2');
T = net_func(w1_rx_data, "W1", "rx", 0, 1, 0);
writematrix(T,result_file,'Sheet',"Network",'Range','K2');
T = net_func(w2_tx_data, "W2", "tx", 0, 1, 0);
writematrix(T,result_file,'Sheet',"Network",'Range','N2');
T = net_func(w2_rx_data, "W2", "rx", 0, 1, 0);
writematrix(T,result_file,'Sheet',"Network",'Range','Q2');

function y = net_func(data, device_type, data_type, peak_threshold, start_rm_peak, end_rm_peak)

% 기본 그래프
[pks,locs] = findpeaks(data ,'MinPeakHeight',peak_threshold);
graph_name = "Network " + device_type + " " + data_type;
figure('NumberTitle', 'off', 'Name', graph_name)

subplot(2,2,1)
findpeaks(data ,'MinPeakHeight',peak_threshold)
text(locs+.02,pks,num2str((1:numel(pks))'))
title(graph_name)
grid on

% 순수 데이터 분포 (히스토그램)
subplot(2,2,2)
nbins = 15;
data_hist = histogram(data(locs(start_rm_peak):locs(size(pks,1)-end_rm_peak)),nbins);hold on;
edges = data_hist.BinEdges;
counts = data_hist.BinCounts;
values = data_hist.Values;
locs(start_rm_peak);
plot(edges(:,1:end-1),values);hold on;
[a,b] = findpeaks([edges(:,1:end-1),values],'SortStr','descend');
grid on
title(graph_name+" 데이터의 분포")

% Peak 개수
temp = size(pks);
peak_num = temp(:,1) - start_rm_peak - end_rm_peak;

% Peak Interval 평균
peakInterval = diff(locs(start_rm_peak+1:start_rm_peak+peak_num));
peakInterval_avg = mean(peakInterval);

% Peak Interval 분포 (히스토그램)
subplot(2,2,3)
peakInterval_hist = histogram(peakInterval);
title("Peak Interval 분포")
grid on

% Peak 값의 분포 (히스토그램)
subplot(2,2,4)
peak_graph = data(locs(start_rm_peak+1:start_rm_peak+peak_num));
peak_hist = histogram(peak_graph);
title("Peak 값의 분포")
grid on

histfit(peak_graph)
pd = fitdist(peak_graph,'Normal')
pd.mu

%total, peakConsumption
total = locs(start_rm_peak+peak_num) - locs(start_rm_peak+1)
peakConsumption = mean(data(locs(start_rm_peak+1:start_rm_peak+peak_num)))
%[pkss,locss] = findpeaks(data,"MinPeakHeight",0);
%dd = sum(peakInterval)-((locss(start_rm_peak+peak_num+1)-locss(start_rm_peak))-peak_num)

%peaktime
xx = locs(start_rm_peak):locs(size(pks,1)-end_rm_peak);
yy = data(locs(start_rm_peak):locs(size(pks,1)-end_rm_peak))
yy = reshape(yy,[1,length(yy)])
dy = diff(yy);
dx = diff(xx);
dy_dx = [0 dy./dx];
startpoint = zeros([peak_num 1]);
endpoint = zeros([peak_num 1]);
for ii = 1:peak_num
    sp = find((xx < locs(start_rm_peak+ii)) & (dy_dx <= 0),1,'last');
    if isempty(sp)
        sp = 1;
    end
    startpoint(ii) = sp;
    ep = find((xx > locs(start_rm_peak+ii)) & (dy_dx >= 0),1,'first');
    if isempty(ep)
        ep = length(xx);
    end
    endpoint(ii) = ep;
end
fprintf("ENDPONT")
peakWidth = xx(endpoint) - xx(startpoint);

%output
y = ["데이터", graph_name; 
    "피크 개수", peak_num; 
    "피크 간격 평균", peakInterval_avg;
    "피크 분포 평균", pd.mu; 
    "피크 분포 분산", pd.sigma;
    "총 시간", total*0.16;
    "피크 시간", mean(peakWidth)*0.16;
    "피크 소비", peakConsumption;
    "유휴 시간", a(1)*0.16*(peak_num-1);
    "유휴 소비", edges(2)/2
    ];

size_b = size(b);

for i = 1:size_b(2)
peak = edges(2) * (b(i) - b(1)) + edges(2)/2;
y = [y; "peak"+i, peak;];  
y = [y; "peak"+i+" 값", a(i);];
end

end


    
