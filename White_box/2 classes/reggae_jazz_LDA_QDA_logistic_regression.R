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

#data generation - REGGAE & JAZZ

n_classes <- 2

reggae_data<- read_csv("reggae.csv")
reggae_data$binary_genre<-0

jazz_data<-read_csv("jazz.csv")
jazz_data$binary_genre<-1

train_data<-rbind(reggae_data[0:80,],jazz_data[0:80,])
test_data<-rbind(reggae_data[81:100,],jazz_data[81:100,])

#priors 
p<-c(1/2,1/2)

#assumptions for qda, lda

#gauss
mcshapiro.test(train_data[which(train_data$binary_genre=="1"),1:7]) 
mcshapiro.test(train_data[which(train_data$binary_genre=="0"),1:7])

# definetly not gaussian


#covariance 
v1<-var(train_data[which(train_data$binary_genre==1),1:7])
v2<-var(train_data[which(train_data$binary_genre==0),1:7])
v1
v2

# homoscedasticity could hold

#QDA (dati gaussiani, no same covariance)
q<-qda(binary_genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast, prior=p, data = train_data)
q #means

#metrics
Qda.m <- predict(object=q, method = "plug-in")
f= factor(train_data$binary_genre)
table(true.lable=f, class.assigned=Qda.m$class)

t_train <- table(true.label = f , assigned.label =Qda.m$class )

Qda.m <- predict(object = q, newdata = data.frame(test_data[,1:7]), method = "plug-in")
f= factor(test_data$binary_genre)
table(true.lable=f, class.assigned=Qda.m$class)

t_test <- table(true.label = f , assigned.label =Qda.m$class )

qda_metrics <- get_metrics_train_test(t_train, t_test, n_classes)
qda_metrics

#           training      test
# accuracy    0.9125 0.6750000
# precision   0.9125 0.6754386
# recall      0.9125 0.6750000
# F1_score    0.9125 0.6747967


#LDA (dati NON gaussiani, same covariance)
l<-lda(binary_genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast,prior=p, data = train_data)
l #means

#metrics
Lda.m <- predict(l, method = "plug-in")
f= factor(train_data$binary_genre)
table(true.lable=f, class.assigned=Lda.m$class)

t_train <- table(true.label = f , assigned.label =Lda.m$class )

Lda.m <- predict(object = l,newdata = test_data, method = "plug-in")
f= factor(test_data$binary_genre)
table(true.lable=f, class.assigned=Lda.m$class)

t_test <- table(true.label = f , assigned.label =Lda.m$class )

lda_metrics <- get_metrics_train_test(t_train, t_test, n_classes)
lda_metrics

#            training      test
# accuracy  0.8875000 0.7750000
# precision 0.8877423 0.7756892
# recall    0.8875000 0.7750000
# F1_score  0.8874824 0.7748593

#could be an overfit, still good performances, but no theoretical background

#logistic regression + covariate selection

glm_model<-glm(binary_genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast ,family=binomial( link = logit ), data = train_data)
summary(glm_model)
pscl::pR2(glm_model)["McFadden"]
# R^2 = 0.6797698

# we apply backward selection: in order we remove spec_flat, lf_contrast, rms_energy, zrc

glm_model_red <-glm(binary_genre~ mean_chroma+hf_contrast+mf_contrast ,family=binomial( link = logit ), data = train_data)
summary(glm_model_red)
pscl::pR2(glm_model_red)["McFadden"]
# R^2 = 0.6681052

#metrics
pred_train <- as.numeric(fitted(object = glm_model_red)>0.5)
f= factor(train_data$binary_genre)
table(true.lable=f, class.assigned=pred_train)

t_train <- table(true.label = f , assigned.label =pred_train )

pred_test <- as.numeric(predict(object = glm_model_red, newdata= test_data, type="response")>0.5)
f= factor(test_data$binary_genre)
table(true.lable=f, class.assigned=pred_test)

t_test <- table(true.label = f , assigned.label =pred_test )

LR_metrics <- get_metrics_train_test(t_train, t_test, n_classes)
LR_metrics

#            training      test
# accuracy  0.8875000 0.7500000
# precision 0.8877423 0.7525253
# recall    0.8875000 0.7500000
# F1_score  0.8874824 0.7493734



#diagnostic
residualPlots(glm_model_red)
