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

blues_data<- read_csv("blues.csv")
blues_data$genre<-"blues"

jazz_data<- read_csv("jazz.csv")
jazz_data$genre<-"jazz"

country_data<- read_csv("country.csv")
country_data$genre<-"country"

reggae_data<-read_csv("reggae.csv")
reggae_data$genre<-"reggae"

hiphop_data<-read_csv("hiphop.csv")
hiphop_data$genre<-"hiphop"

train_data<-rbind(blues_data[0:80,],jazz_data[0:80,],country_data[0:80,],reggae_data[0:80,],hiphop_data[0:80,])
test_data<-rbind(blues_data[81:100,],jazz_data[81:100,],country_data[81:100,],reggae_data[81:100,],hiphop_data[81:100,])


#priors 
p<-rep(1/5,5)


#gauss
mcshapiro.test(train_data[which(train_data$genre=="blues"),1:7])
mcshapiro.test(train_data[which(train_data$genre=="jazz"),1:7])
mcshapiro.test(train_data[which(train_data$genre=="country"),1:7])
mcshapiro.test(train_data[which(train_data$genre=="reggae"),1:7])
mcshapiro.test(train_data[which(train_data$genre=="hiphop"),1:7])

#definetly not gaussian

#covariance 
v1<-var(train_data[which(train_data$genre=="blues"),1:7])
v2<-var(train_data[which(train_data$genre=="jazz"),1:7])
v3<-var(train_data[which(train_data$genre=="country"),1:7])
v4<-var(train_data[which(train_data$genre=="reggae"),1:7])
v5<-var(train_data[which(train_data$genre=="hiphop"),1:7])
v1
v2
v3
v4
v5

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

#            training      test
# accuracy  0.6575000 0.4800000
# precision 0.6860550 0.4772844
# recall    0.6575000 0.4800000
# F1_score  0.6504339 0.4475484


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
# accuracy  0.5925000 0.2600000
# precision 0.5901032 0.2437274
# recall    0.5925000 0.2600000
# F1_score  0.5880321       NaN

# multinomial logistic regression

model <- nnet::multinom(genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast, data = train_data)
summary(model)
pscl::pR2(model)["McFadden"]
# R^2 = 0.432471 

#we reduce the model based on approximate confidence intervals, applying backward selection we remove hf_contrast
model_red <- nnet::multinom(genre~ zcr+rms_energy+mean_chroma+spec_flat+mf_contrast+lf_contrast, data = train_data)
summary(model_red)
pscl::pR2(model_red)["McFadden"]
# R^2 = 0.4139059  
#unsure, but zcr could be reduced
model_red2 <- nnet::multinom(genre~ rms_energy+mean_chroma+spec_flat+mf_contrast+lf_contrast, data = train_data)
summary(model_red2)
pscl::pR2(model_red2)["McFadden"]
# R^2 = 0.4028664  


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
# accuracy  0.5925000 0.2600000
# precision 0.5901032 0.2437274
# recall    0.5925000 0.2600000
# F1_score  0.5880321       NaN
#####

# second reduced model
#####
#accuracy on training data:
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

#            training      test
# accuracy  0.5850000 0.3900000
# precision 0.5775952 0.3746702
# recall    0.5850000 0.3900000
# F1_score  0.5765480       NaN
#####
