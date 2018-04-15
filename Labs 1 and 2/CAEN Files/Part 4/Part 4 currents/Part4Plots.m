% Lab 2 Part 4 Model Graphs
twelve = importdata('12um/currents')
twentyfour = importdata('24um/currents')
fourtyeight = importdata('48um/currents')

timeaxis = twelve.data(100:1203, 1);
hillock12 = twelve.data(100:1203, 205);
hillock24 = twentyfour.data(100:1203, 205);
hillock48 = fourtyeight.data(100:1203, 205);

plot(timeaxis, hillock12)
hold on
plot(timeaxis, hillock24)
plot(timeaxis, hillock48)