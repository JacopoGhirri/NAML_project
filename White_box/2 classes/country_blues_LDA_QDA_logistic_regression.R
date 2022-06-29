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


#data generation - Country & Blues

n_classes <- 2

country_data<- read_csv("country.csv")
country_data$binary_genre<-0

blues_data<-read_csv("blues.csv")
blues_data$binary_genre<-1

train_data<-rbind(country_data[0:80,],blues_data[0:80,])
test_data<-rbind(country_data[81:100,],blues_data[81:100,])

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

# clearly heteroscedastic

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

# training      test
# accuracy  0.8437500 0.6750000
# precision 0.8747232 0.8030303
# recall    0.8437500 0.6750000
# F1_score  0.8404531 0.6366177


#LDA (dati NON gaussiani, same covariance)
l<-lda(binary_genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast,prior=p, data = train_data)
l #means

#aper_train data
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

# training      test
# accuracy  0.7687500 0.4000000
# precision 0.7848154 0.2222222
# recall    0.7687500 0.4000000
# F1_score  0.7654424       NaN

#quite poor performances, it is possible that the selected features do not provide a sufficient separation

#logistic regression + covariate selection

glm_model<-glm(binary_genre~ zcr+rms_energy+mean_chroma+spec_flat+hf_contrast+mf_contrast+lf_contrast ,family=binomial( link = logit ), data = train_data)
summary(glm_model)
pscl::pR2(glm_model)["McFadden"]
# R^2 = 0.412

# we apply backward selection: in order we remove mf_contrast, mean_chroma, lf_contrast

glm_model_red <-glm(binary_genre~ zcr+rms_energy+spec_flat+hf_contrast,family=binomial( link = logit ), data = train_data)
summary(glm_model_red)
pscl::pR2(glm_model_red)["McFadden"]
# R^2 = 0.397, not really good

#metrics:
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

# training      test
# accuracy  0.7812500 0.4750000
# precision 0.7914980 0.4641577
# recall    0.7812500 0.4750000
# F1_score  0.7793103 0.4320487


#diagnostic
residualPlots(glm_model_red)
