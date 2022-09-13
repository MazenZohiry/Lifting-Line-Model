clc; clear; close all;
%% data
chord=csvread('Optimized_chord.csv');
twist=csvread('Optimized_twist.csv');
r_R = linspace(0.2,1,length(chord));

figure(1)
plot(r_R.', chord)
xlabel('r/R') 
ylabel('chord distribution') 
grid on
grid minor
figure(2)
plot(r_R.', twist)
xlabel('r/R') 
ylabel('twsit distribution') 
grid on
grid minor
