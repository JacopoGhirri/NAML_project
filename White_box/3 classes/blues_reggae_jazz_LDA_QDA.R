setwd("C:/Users/user/Desktop/universitÓ/dare/Numerical Analysis for Machine Learning/NAML proj/NAML_repo/NAML_project/White_box/Dati")
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

source("C:/Users/user/Desktop/universitÓ/dare/Numerical Analysis for Machine Learning/NAML proj/NAML_repo/NAML_project/White_box/metric_extractor.R")


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
jazz_data$genre <- as.factor(jazz_data$genre)

blues_data<-read_csv("blues.csv")
blues_data$genre<-"blues"
blues_data$genre <- as.factor(blues_data$genre)

reggae_data<-read_csv("reggae.csv")
reggae_data$genre<-"reggae"
reggae_data$genre <- as.factor(reggae_data$genre)

train_data<-rbind(jazz_data[0:80,],blues_data[0:80,],reggae_data[0:80,])
test_data<-rbind(jazz_data[81:100,],blues_data[81:100,],reggae_data[81:100,])

genres <- c("jazz", "blues", "reggae")

#priors 
p<-rep(1/3,3)

#assumptions

#gauss
mcshapiro.test(train_data[which(train_data$genre=="blues"),1:7])
mcshapiro.test(train_data[which(train_data$genre=="jazz"),1:7])
mcshapiro.test(train_data[which(train_data$genre=="reggae"),1:7])

#definetly not gaussian

#covariance 
v1<-var(train_data[which(train_data$genre=="blues"),1:7])
v2<-var(train_data[which(train_data$genre=="jazz"),1:7])
v3<-var(train_data[which(train_data$genre=="reggae"),1:7])
v1
v2
v3

#not homoscedasticity


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

#            training      test
# accuracy  0.7833333 0.6833333
# precision 0.8219260 0.6664863
# recall    0.7833333 0.6833333
# F1_score  0.7818805 0.6669638


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
# accuracy  0.7458333 0.7500000
# precision 0.7496077 0.7884615
# recall    0.7458333 0.7500000
# F1_score  0.7460626 0.7563939

# multinomial logistic regression

model <- nnet::multinom(genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast, data = train_data)
summary(model)
pscl::pR2(model)["McFadden"]
# R^2 = 0.5369562

#we reduce the model based on approximate confidence intervals, applying backward selection we remove zrc
model_red <- nnet::multinom(genre~ rms_energy+mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast, data = train_data)
summary(model_red)
pscl::pR2(model_red)["McFadden"]
# R^2 = 0.5265427
#unsure, but rms_energy could be reduced
model_red2 <- nnet::multinom(genre~ mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast, data = train_data)
summary(model_red2)
pscl::pR2(model_red2)["McFadden"]
# R^2 = 0.4831856


# first reduced model
#####
#metrics:
pred_train <- factor(genres[argmax(fitted(object = model_red))])
f= factor(train_data$genre)
table(true.lable=f, class.assigned=pred_train)

t_train <- table(true.label = f , assigned.label =pred_train )
t_train[,c(1,2)] <- t_train[,c(2,1)] #reordering

pred_test <- factor(predict(object = model_red, newdata= test_data, type="class"))
f= factor(test_data$genre)
table(true.lable=f, class.assigned=pred_test)

t_test <- table(true.label = f , assigned.label =pred_test )

MR_metrics <- get_metrics_train_test(t_train, t_test, n_classes)
MR_metrics

#            training      test
# accuracy  0.7416667 0.7000000
# precision 0.7426017 0.7368814
# recall    0.7416667 0.7000000
# F1_score  0.7420055 0.7051051
#####

# second reduced model
#####
#accuracy on training data:
pred_train <- factor(genres[argmax(fitted(object = model_red2))])
f= factor(train_data$genre)
table(true.lable=f, class.assigned=pred_train)

t_train <- table(true.label = f , assigned.label =pred_train )
t_train[,c(1,2)] <- t_train[,c(2,1)] #reordering

pred_test <- factor(predict(object = model_red2, newdata= test_data, type="class"))
f= factor(test_data$genre)
table(true.lable=f, class.assigned=pred_test)

t_test <- table(true.label = f , assigned.label =pred_test )

MR_metrics <- get_metrics_train_test(t_train, t_test, n_classes)
MR_metrics

#            training      test
# accuracy  0.7041667 0.7000000
# precision 0.7145697 0.7244916
# recall    0.7041667 0.7000000
# F1_score  0.7069249 0.7024258
#####
