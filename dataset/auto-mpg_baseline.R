setwd('E:/Dropbox/2019-202x study (phd)/yr2 sem2/internship/code/dataset')
dat = read.csv('auto-mpg.csv')

dat['horsepower1'] = 0
dat['horsepower1'][dat['horsepower'] == '?'] = round(mean(as.numeric(dat['horsepower'][dat['horsepower'] != '?'])))
dat['horsepower1'][dat['horsepower'] != '?'] = as.numeric(dat['horsepower'][dat['horsepower'] != '?'])

set.seed(1)
ind = sample(nrow(dat), nrow(dat))
dat_train = dat[ind[1:300], ]
dat_test = dat[ind[301:nrow(dat)], ]

model = lm(mpg ~ cylinders + displacement + weight + acceleration + model.year + origin + horsepower1, data = dat_train)

mean(abs((predict(model, newdata = dat_train) - dat_train$mpg) / dat_train$mpg))
#[1] 0.1163063
mean(abs((predict(model, newdata = dat_test) - dat_test$mpg) / dat_test$mpg))
#[1] 0.1266464