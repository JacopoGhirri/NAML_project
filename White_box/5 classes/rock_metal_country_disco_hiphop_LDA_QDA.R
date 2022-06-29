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

n_classes <- 5

rock_data<- read_csv("rock.csv")
rock_data$genre<-"rock"

metal_data<- read_csv("metal.csv")
metal_data$genre<-"metal"

country_data<- read_csv("country.csv")
country_data$genre<-"country"

disco_data<-read_csv("disco.csv")
disco_data$genre<-"disco"

hiphop_data<-read_csv("hiphop.csv")
hiphop_data$genre<-"hiphop"

train_data<-rbind(rock_data[0:80,],metal_data[0:80,],country_data[0:80,],disco_data[0:80,],hiphop_data[0:80,])
test_data<-rbind(rock_data[81:100,],metal_data[81:100,],country_data[81:100,],disco_data[81:100,],hiphop_data[81:100,])


#priors 
p<-rep(1/5,5)

#assumptions

#gauss
mcshapiro.test(data[which(genre=="rock"),1:7])
mcshapiro.test(data[which(genre=="metal"),1:7])
mcshapiro.test(data[which(genre=="country"),1:7])
mcshapiro.test(data[which(genre=="disco"),1:7])
mcshapiro.test(data[which(genre=="hiphop"),1:7])

#covariance 
v1<-var(data[which(genre=="country"),1:7])
v2<-var(data[which(genre=="disco"),1:7])
v3<-var(data[which(genre=="hiphop"),1:7])
v4<-var(data[which(genre=="rock"),1:7])
v5<-var(data[which(genre=="metal"),1:7])
v1
v2
v3
v4
v5



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
# accuracy  0.6475000 0.4300000
# precision 0.6598780 0.4895873
# recall    0.6475000 0.4300000
# F1_score  0.6326121 0.3955583


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
# accuracy  0.6250000 0.4100000
# precision 0.6134596 0.4493066
# recall    0.6250000 0.4100000
# F1_score  0.6136527 0.3994019

# multinomial logistic regression

model <- nnet::multinom(genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast, data = train_data)
summary(model)
pscl::pR2(model)["McFadden"]
# R^2 = 0.4133652

#we reduce the model based on approximate confidence intervals, applying backward selection we remove rms_energy, mf_contrast, hf_contast
model_red <- nnet::multinom(genre~ zcr+mean_chroma+spec_flat+lf_contrast, data = train_data)
summary(model_red)
pscl::pR2(model_red)["McFadden"]
# R^2 = 0.3640281  

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
# accuracy  0.6025000 0.4400000
# precision 0.5853185 0.4948485
# recall    0.6025000 0.4400000
# F1_score  0.5893005 0.4403928
