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

source("C:/Users/user/Desktop/università/dare/Numerical Analysis for Machine Learning/NAML proj/NAML_repo/NAML_project/White_box/metric_extractor.R")

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

n_classes <- 3

jazz_data<- read_csv("jazz.csv")
jazz_data$genre<-"jazz"

classical_data<-read_csv("classical.csv")
classical_data$genre<-"classical"

rock_data<-read_csv("rock.csv")
rock_data$genre<-"rock"


train_data<-rbind(jazz_data[0:80,],classical_data[0:80,],rock_data[0:80,])
test_data<-rbind(jazz_data[81:100,],classical_data[81:100,],rock_data[81:100,])

genres <- c("jazz", "classical", "rock")

#priors 
p<-rep(1/3,3)


#gauss
mcshapiro.test(train_data[which(train_data$genre=="jazz"),1:7])
mcshapiro.test(train_data[which(train_data$genre=="classical"),1:7])
mcshapiro.test(train_data[which(train_data$genre=="rock"),1:7])

#definetly not gaussian

#covariance 
v1<-var(train_data[which(train_data$genre=="jazz"),1:7])
v2<-var(train_data[which(train_data$genre=="classical"),1:7])
v3<-var(train_data[which(train_data$genre=="rock"),1:7])
v1
v2
v3

# not homoscedastic


#QDA (dati gaussiani, no same covariance)
q<-qda(genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast, prior=p, data = train_data)
q #means

#metrics
Qda.m <- predict(object=q, method = "plug-in")
f= factor(train_data$genre)
table(true.lable=f, class.assigned=Qda.m$class)

t_train <- table(true.label = f , assigned.label =Qda.m$class )

Qda.m <- predict(object = q, newdata = data.frame(test_data[,1:7]), method = "plug-in")
f= factor(test_data$genre)
table(true.lable=f, class.assigned=Qda.m$class)

t_test <- table(true.label = f , assigned.label =Qda.m$class )

qda_metrics <- get_metrics_train_test(t_train, t_test, n_classes)
qda_metrics

#            training     test
# accuracy  0.8125000 0.650000
# precision 0.8120722 0.654321
# recall    0.8125000 0.650000
# F1_score  0.8116444 0.599018


#LDA (dati NON gaussiani, same covariance)
l<-lda(genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast,prior=p, data = train_data)
l #means

#metrics
Lda.m <- predict(l, method = "plug-in")
f= factor(train_data$genre)
table(true.lable=f, class.assigned=Lda.m$class)

t_train <- table(true.label = f , assigned.label =Lda.m$class )

Lda.m <- predict(object = l,newdata = test_data, method = "plug-in")
f= factor(test_data$genre)
table(true.lable=f, class.assigned=Lda.m$class)

t_test <- table(true.label = f , assigned.label =Lda.m$class )

lda_metrics <- get_metrics_train_test(t_train, t_test, n_classes)
lda_metrics

#            training      test
# accuracy  0.7708333 0.7166667
# precision 0.7690946 0.7475580
# recall    0.7708333 0.7166667
# F1_score  0.7697066 0.6674563

# multinomial logistic regression

model <- nnet::multinom(genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast, data = train_data)
summary(model)
pscl::pR2(model)["McFadden"]
# R^2 = 0.5310116

#we reduce the model based on approximate confidence intervals, applying backward selection we remove mf_contrast
model_red <- nnet::multinom(genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast+lf_contrast, data = train_data)
summary(model_red)
pscl::pR2(model_red)["McFadden"]
# R^2 = 0.5097416 
#unsure, but hf_contrast could be reduced
model_red2 <- nnet::multinom(genre~ zcr+rms_energy+mean_chroma+spec_flat+lf_contrast, data = train_data)
summary(model_red2)
pscl::pR2(model_red2)["McFadden"]
# R^2 = 0.493429 


# first reduced model
#####
#metrics
pred_train <- factor(predict(object = model_red,type="class"))
f= factor(train_data$genre)
table(true.lable=f, class.assigned=pred_train)

t_train <- table(true.label = f , assigned.label =pred_train )

pred_test <- factor(predict(object = model_red, newdata= test_data, type="class"))
f= factor(test_data$genre)
table(true.lable=f, class.assigned=pred_test)

t_test <- table(true.label = f , assigned.label =pred_test )

MR_metrics <- get_metrics_train_test(t_train, t_test, n_classes)
MR_metrics

#            training      test
# accuracy  0.7625000 0.7166667
# precision 0.7614551 0.7469136
# recall    0.7625000 0.7166667
# F1_score  0.7619003 0.6672122
#####

# second reduced model
#####
#metrics
pred_train <- factor(predict(object = model_red2,type="class"))
f= factor(train_data$genre)
table(true.lable=f, class.assigned=pred_train)

t_train <- table(true.label = f , assigned.label =pred_train )

pred_test <- factor(predict(object = model_red2, newdata= test_data, type="class"))
f= factor(test_data$genre)
table(true.lable=f, class.assigned=pred_test)

t_test <- table(true.label = f , assigned.label =pred_test )

MR_metrics <- get_metrics_train_test(t_train, t_test, n_classes)
MR_metrics

#           training      test
# accuracy  0.7750000 0.6833333
# precision 0.7734081 0.7246743
# recall    0.7750000 0.6833333
# F1_score  0.7739490 0.6397698
#####
