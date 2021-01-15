% 경로 및 읽어올 파일, 결과 저장 파일 설정
folder_path = "";
fileID = fopen(folder_path+"model_sync_ps_cpu.txt",'r');
C = textscan(fileID,'%d %s %f %f %s %s %f %s %f %f %s %s')
ps_cpu_data = C{9};
fileID = fopen(folder_path+"model_sync_w1_cpu.txt",'r');
C = textscan(fileID,'%d %s %f %f %s %s %f %s %f %f %s %s')
w1_cpu_data = C{9};
fileID = fopen(folder_path+"model_sync_w2_cpu.txt",'r');
C = textscan(fileID,'%d %s %f %f %s %s %f %s %f %f %s %s')
w2_cpu_data = C{9};
result_file = folder_path+"model_sync_Matlab_result.xlsx";

% 함수 / 데이터 종류 3개 (PS, W1, W2)
% cpu_func(data, device_type, peak_threshold, start_rm_peak, end_rm_peak)
T = cpu_func(ps_cpu_data, "PS", 0, 1, 0);
writematrix(T,result_file,'Sheet','CPU','Range','B2');
T = cpu_func(w1_cpu_data, "W1", 0, 1, 0);
writematrix(T,result_file,'Sheet','CPU','Range','E2');
T = cpu_func(w2_cpu_data, "W2", 0, 1, 0);
writematrix(T,result_file,'Sheet','CPU','Range','H2');

function y = cpu_func(data, device_type, peak_threshold, start_rm_peak, end_rm_peak)

% 기본 그래프
[pks,locs] = findpeaks(data ,'MinPeakHeight',peak_threshold);
graph_name = "CPU " + device_type + " utilization";
figure('NumberTitle', 'off', 'Name', graph_name)

subplot(2,2,1)
findpeaks(data ,'MinPeakHeight',peak_threshold);
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
title("Peak 값의 분포")
grid on

histfit(peak_graph);
pd = fitdist(peak_graph,'Normal');
pd.mu;

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


    