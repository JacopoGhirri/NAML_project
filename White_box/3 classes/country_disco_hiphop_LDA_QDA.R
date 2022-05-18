setwd("C:/Users/user/Desktop/università/dare/Numerical Analysis for Machine Learning/NAML proj/NAML_repo/NAML_project/White_box/Dati")
#####
library(MASS)
library(class)

library(mvtnorm)
library(mvnormtest)
library(car)
library(MASS)
library(class)
library(readr)
library(pscl)


library(rms)
library(arm)
library(ResourceSelection)
library(pROC)

library(tidyverse)
library(caret)
library(nnet)
library(ramify)

mcshapiro.test <- function(X, devstmax = 0.01, sim = ceiling(1/(4*devstmax^2)))
{
  library(mvnormtest)
  n   <- dim(X)[1]
  p   <- dim(X)[2]
  mu  <- rep(0,p)
  sig <- diag(p)
  W   <- NULL
  for(i in 1:sim)
  {
    Xsim <- rmvnorm(n, mu, sig)
    W   <- c(W, mshapiro.test(t(Xsim))$stat)
    # mshapiro.test(X): compute the statistics min(W) for the sample X
  }
  Wmin   <- mshapiro.test(t(X))$stat   # min(W) for the given sample
  pvalue <- sum(W < Wmin)/sim          # proportion of min(W) more extreme than the observed Wmin
  devst  <- sqrt(pvalue*(1-pvalue)/sim)
  list(Wmin = as.vector(Wmin), pvalue = pvalue, devst = devst, sim = sim)
}
#####

#data generation

country_data<- read_csv("country.csv")
country_data$genre<-"country"

disco_data<-read_csv("disco.csv")
disco_data$genre<-"disco"

hiphop_data<-read_csv("hiphop.csv")
hiphop_data$genre<-"hiphop"


train_data<-rbind(country_data[0:80,],disco_data[0:80,],hiphop_data[0:80,])
test_data<-rbind(country_data[81:100,],disco_data[81:100,],hiphop_data[81:100,])

genres <- c("country", "disco", "hiphop")

#priors 
p<-rep(1/3,3)


#gauss
mcshapiro.test(train_data[which(train_data$genre=="country"),1:7])
mcshapiro.test(train_data[which(train_data$genre=="disco"),1:7])
mcshapiro.test(train_data[which(train_data$genre=="hiphop"),1:7])

#definetly not gaussian

#covariance 
v1<-var(train_data[which(train_data$genre=="country"),1:7])
v2<-var(train_data[which(train_data$genre=="disco"),1:7])
v3<-var(train_data[which(train_data$genre=="hiphop"),1:7])
v1
v2
v3

#could be homoscedastic


#QDA (dati gaussiani, no same covariance)
q<-qda(genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast, prior=p, data = train_data)
q #means

#aper_train data
Qda.m <- predict(object=q, method = "plug-in")
f= factor(train_data$genre)
table(true.lable=f, class.assigned=Qda.m$class)

l <-length(levels(as.factor(f))) 
t <- table(true.label = f , assigned.label =Qda.m$class )
train_APER_qda <- 0
for(i in 1:l){
  train_APER_qda <- train_APER_qda + sum(t[i,-i])*p[i]/sum(t[i,])
}
train_APER_qda

#aper_test data
Qda.m <- predict(object = q, newdata = data.frame(test_data[,1:7]), method = "plug-in")
f= factor(test_data$genre)
table(true.lable=f, class.assigned=Qda.m$class)

l <-length(levels(as.factor(f))) 
t <- table(true.label = f , assigned.label =Qda.m$class )
test_APER_qda <- 0
for(i in 1:l){
  test_APER_qda <- test_APER_qda + sum(t[i,-i])*p[i]/sum(t[i,])
}
test_APER_qda

qda_accuracies = cbind(training = 1-train_APER_qda,test = 1-test_APER_qda)
qda_accuracies
# training  test
# 0.8       0.8333333


#LDA (dati NON gaussiani, same covariance)
l<-lda(genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast,prior=p, data = train_data)
l #means

#aper_train data
Lda.m <- predict(l, method = "plug-in")
f= factor(train_data$genre)
table(true.lable=f, class.assigned=Lda.m$class)

len <-length(levels(as.factor(f))) 
t <- table(true.label = f , assigned.label =Lda.m$class )
train_APER_lda <- 0
for(i in 1:len){
  train_APER_lda <- train_APER_lda + sum(t[i,-i])*p[i]/sum(t[i,])
}
train_APER_lda

#aper_test data
Lda.m <- predict(object = l,newdata = test_data, method = "plug-in")
f= factor(test_data$genre)
table(true.lable=f, class.assigned=Lda.m$class)

len <-length(levels(as.factor(f))) 
t <- table(true.label = f , assigned.label =Lda.m$class )
test_APER_lda <- 0
for(i in 1:len){
  test_APER_lda <- test_APER_lda + sum(t[i,-i])*p[i]/sum(t[i,])
}
test_APER_lda

lda_accuracies = cbind(training = 1-train_APER_lda,test = 1-test_APER_lda)
lda_accuracies
# training  test
# 0.7708333 0.7333333

# multinomial logistic regression

model <- nnet::multinom(genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast, data = train_data)
summary(model)
pscl::pR2(model)["McFadden"]
# R^2 = 0.4366478

#we reduce the model based on approximate confidence intervals, applying backward selection we remove hf_contrast, rms_energy, mf_contrast
model_red <- nnet::multinom(genre~ zcr+mean_chroma+spec_flat+lf_contrast, data = train_data)
summary(model_red)
pscl::pR2(model_red)["McFadden"]
# R^2 = 0.3891507
#unsure, but lf_contrast could be reduced
model_red2 <- nnet::multinom(genre~ zcr+mean_chroma+spec_flat, data = train_data)
summary(model_red2)
pscl::pR2(model_red2)["McFadden"]
# R^2 = 0.3806529


# first reduced model
#####
#accuracy on training data:
pred_train <- factor(genres[argmax(fitted(object = model_red))])
f= factor(train_data$genre)
table(true.lable=f, class.assigned=pred_train)

len <-length(levels(as.factor(f))) 
t <- table(true.label = f , assigned.label =pred_train )
train_acc_logit <- mean(f == pred_train)
train_acc_logit
#accuracy on test data:
pred_test <- factor(predict(object = model_red, newdata= test_data, type="class"))
f= factor(test_data$genre)
table(true.lable=f, class.assigned=pred_test)

len <-length(levels(as.factor(f))) 
t <- table(true.label = f , assigned.label =pred_test )
test_acc_logit <- mean(f == pred_test)
test_acc_logit

logistic_regression_accuracies = cbind(training = train_acc_logit, test = test_acc_logit)
logistic_regression_accuracies
# training  test
# 0.7541667 0.7166667
#####

# second reduced model
#####
#accuracy on training data:
pred_train <- factor(genres[argmax(fitted(object = model_red2))])
f= factor(train_data$genre)
table(true.lable=f, class.assigned=pred_train)

len <-length(levels(as.factor(f))) 
t <- table(true.label = f , assigned.label =pred_train )
train_acc_logit <- mean(f == pred_train)
train_acc_logit
#accuracy on test data:
pred_test <- factor(predict(object = model_red2, newdata= test_data, type="class"))
f= factor(test_data$genre)
table(true.lable=f, class.assigned=pred_test)

len <-length(levels(as.factor(f))) 
t <- table(true.label = f , assigned.label =pred_test )
test_acc_logit <- mean(f == pred_test)
test_acc_logit

logistic_regression_accuracies = cbind(training = train_acc_logit, test = test_acc_logit)
logistic_regression_accuracies
# training  test
# 0.7375    0.75
#####
