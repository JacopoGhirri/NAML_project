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

n_classes <- 10

blues_data<- read_csv("blues.csv")
blues_data$genre<-"blues"

classical_data<- read_csv("classical.csv")
classical_data$genre<-"classical"

jazz_data<- read_csv("jazz.csv")
jazz_data$genre<-"jazz"

country_data<- read_csv("country.csv")
country_data$genre<-"country"

reggae_data<-read_csv("reggae.csv")
reggae_data$genre<-"reggae"

hiphop_data<-read_csv("hiphop.csv")
hiphop_data$genre<-"hiphop"

disco_data<- read_csv("disco.csv")
disco_data$genre<-"disco"

metal_data<- read_csv("metal.csv")
metal_data$genre<-"metal"

pop_data<- read_csv("pop.csv")
pop_data$genre<-"pop"

rock_data<- read_csv("rock.csv")
rock_data$genre<-"rock"

train_data<-rbind(blues_data[0:80,],classical_data[0:80,],country_data[0:80,],disco_data[0:80,],hiphop_data[0:80,],jazz_data[0:80,],metal_data[0:80,],pop_data[0:80,],reggae_data[0:80,],rock_data[0:80,])
test_data<-rbind(blues_data[81:100,],classical_data[81:100,],country_data[81:100,],disco_data[81:100,],hiphop_data[81:100,],jazz_data[81:100,],metal_data[81:100,],pop_data[81:100,],reggae_data[81:100,],rock_data[81:100,])


#priors 
p<-rep(1/10,10)


#gauss
genres <- levels(as.factor(train_data$genre))
for(i in 1:10){
  print(mcshapiro.test(train_data[which(train_data$genre==genres[i]),1:7])$pvalue)
}
  
#definetly not gaussian

#covariance 
for(i in 1:10){
  print(var(train_data[which(train_data$genre==genres[i]),1:7]))
}

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
# accuracy  0.5475000 0.3700000
# precision 0.5782914 0.3168942
# recall    0.5475000 0.3700000
# F1_score  0.5193343       NaN


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
# accuracy  0.5200000 0.3350000
# precision 0.5109265 0.2886057
# recall    0.5200000 0.3350000
# F1_score  0.5060305       NaN

# multinomial logistic regression

model <- nnet::multinom(genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast, data = train_data)
summary(model)
pscl::pR2(model)["McFadden"]
# R^2 = 0.4357057 

#we reduce the model based on approximate confidence intervals, applying backward selection we remove lf_contrast, mf_contrast
model_red <- nnet::multinom(genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast, data = train_data)
summary(model_red)
pscl::pR2(model_red)["McFadden"]
# R^2 = 0.4053151   

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

#            training     test
# accuracy  0.5087500 0.345000
# precision 0.4969623 0.330741
# recall    0.5087500 0.345000
# F1_score  0.4988976      NaN
